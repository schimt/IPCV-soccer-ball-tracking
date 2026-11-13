"""Concurrent, CRC-checked member downloads from a pinned SoccerNet train archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
import struct
import time
import zipfile
import zlib

from .common import save_json
from .download import HTTPRangeReader, REPO


def unpack_member(blob, member):
    if len(blob) < 30 or blob[:4] != b'PK\x03\x04':
        raise ValueError('Invalid ZIP local header')
    fields = struct.unpack('<4s5H3I2H', blob[:30])
    flags, method, name_len, extra_len = fields[2], fields[3], fields[-2], fields[-1]
    name = blob[30:30 + name_len].decode('utf-8' if flags & 0x800 else 'cp437')
    if name != member.filename or flags & 1 or method != member.compress_type:
        raise ValueError('Unexpected or encrypted ZIP member')
    start = 30 + name_len + extra_len
    data = blob[start:start + member.compress_size]
    if len(data) != member.compress_size:
        raise ValueError('Truncated member')
    if method == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif method != zipfile.ZIP_STORED:
        raise ValueError('Unsupported compression')
    if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
        raise ValueError('ZIP size or CRC mismatch')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision', required=True)
    p.add_argument('--sequences', nargs='+', required=True)
    p.add_argument('--stride', type=int, required=True)
    p.add_argument('--max-frame', type=int, default=750)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', required=True)
    args = p.parse_args()
    if args.stride < 1 or args.max_frame < 1 or not 1 <= args.workers <= 12:
        raise ValueError('Invalid subset or concurrency')
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        raise ValueError('Use an immutable commit SHA')
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(REPO, 'train.zip', repo_type='dataset', revision=args.revision)
    reader = HTTPRangeReader(url)
    with zipfile.ZipFile(reader) as archive:
        members = sorted(archive.infolist(), key=lambda m: m.header_offset)
        end = archive.start_dir
    ends = {m.header_offset: members[i+1].header_offset if i+1 < len(members) else end
            for i, m in enumerate(members)}
    selected = []
    for m in members:
        parts = PurePosixPath(m.filename).parts
        if not set(args.sequences).intersection(parts) or m.is_dir():
            continue
        f = PurePosixPath(m.filename)
        if f.suffix.lower() == '.jpg':
            if int(f.stem) > args.max_frame or (int(f.stem)-1) % args.stride:
                continue
        elif f.name not in {'seqinfo.ini', 'gameinfo.ini', 'gt.txt'}:
            continue
        selected.append(m)
    found = {part for m in selected for part in PurePosixPath(m.filename).parts}
    if not set(args.sequences).issubset(found):
        raise ValueError('Requested sequence missing from train.zip')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def fetch(member):
        target = root.joinpath(*PurePosixPath(member.filename).parts).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Unsafe archive path')
        if target.exists() and target.stat().st_size == member.file_size:
            if zlib.crc32(target.read_bytes()) == member.CRC:
                return member.filename
        size = ends[member.header_offset] - member.header_offset
        if not 0 < size <= member.compress_size + 200000:
            raise ValueError('Unexpected member span')
        for attempt in range(3):
            try:
                # Each worker gets independent seek state; constructor's HEAD probe is reused.
                stream = HTTPRangeReader.__new__(HTTPRangeReader)
                stream.url, stream.size, stream.pos = url, reader.size, member.header_offset
                data = unpack_member(stream.read(size), member)
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(target.suffix + '.part')
                tmp.write_bytes(data)
                tmp.replace(target)
                return member.filename
            except (OSError, ValueError, RuntimeError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    completed = []
    print(f'Selected {len(selected)} members from pinned train.zip', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch, member) for member in selected]
        for future in as_completed(futures):
            completed.append(future.result())
            if len(completed) % 100 == 0:
                print(f'{len(completed)}/{len(selected)} verified', flush=True)
    save_json(args.manifest, dict(repo=REPO, revision=args.revision, archive='train.zip',
              sequences=args.sequences, stride=args.stride, max_frame=args.max_frame,
              members=sorted(completed), crc_verified=True))
    print('Completed', args.manifest, flush=True)


if __name__ == '__main__':
    main()
