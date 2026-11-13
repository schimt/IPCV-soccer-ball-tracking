"""Read selected ZIP members using HTTP ranges; never download all splits implicitly."""
import argparse
import io
from pathlib import Path, PurePosixPath
import re
import urllib.request
import zipfile
import zlib

from .common import save_json

REPO = 'SoccerNet/SN-Tracking-2023'


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 206:
                raise RuntimeError('Server lacks range support; use hf download for an explicit full ZIP.')
            self.size = int(response.headers['Content-Range'].split('/')[-1])
            response.read()

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if self.pos < 0:
            raise ValueError('Negative seek')
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b''
        # Query also avoids intermediary caches reusing a different byte range.
        url = self.url + ('&' if '?' in self.url else '?') + f'range_start={self.pos}&range_n={n}'
        req = urllib.request.Request(url, headers={'Range': f'bytes={self.pos}-{self.pos+n-1}'})
        with urllib.request.urlopen(req, timeout=90) as response:
            expected = f'bytes {self.pos}-{self.pos+n-1}/{self.size}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise RuntimeError('Server returned an unexpected byte range')
            data = response.read(n)
        if len(data) != n:
            raise IOError('Truncated download')
        self.pos += len(data)
        return data


def main(args):
    if args.stride < 1 or args.max_frame < 0:
        raise ValueError('stride must be positive and max-frame nonnegative')
    from huggingface_hub import HfApi, hf_hub_url
    revision = args.revision or HfApi().dataset_info(REPO).sha
    url = hf_hub_url(REPO, args.archive, repo_type='dataset', revision=revision)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(HTTPRangeReader(url)) as z:
        names = z.namelist()
        sequences = sorted({m.group() for name in names if (m := re.search(r'SNMOT-\d+', name))})
        save_json(out / 'remote_inventory.json', dict(repo=REPO, revision=revision,
                  archive=args.archive, sequences=sequences, example_members=names[:30]))
        print('Revision:', revision, '\nSequences:', ', '.join(sequences), flush=True)
        selected = args.sequences or []
        if args.list:
            return
        if not selected or not set(selected).issubset(sequences):
            raise ValueError('Choose existing --sequences from --list')
        downloaded = []
        for name in names:
            parts = PurePosixPath(name).parts
            seq = next((s for s in selected if s in parts), None)
            if seq is None or name.endswith('/'):
                continue
            is_image = PurePosixPath(name).suffix.lower() in {'.jpg', '.png', '.jpeg'}
            if is_image:
                frame = int(PurePosixPath(name).stem)
                if args.metadata_only or (frame - 1) % args.stride or (args.max_frame and frame > args.max_frame):
                    continue
            elif PurePosixPath(name).name not in {'seqinfo.ini', 'gameinfo.ini', 'gt.txt'}:
                continue
            target = (out / Path(*parts)).resolve()
            if not target.is_relative_to(out.resolve()):
                raise ValueError('Unsafe ZIP path')
            target.parent.mkdir(parents=True, exist_ok=True)
            member = z.getinfo(name)
            valid_existing = (target.exists() and target.stat().st_size == member.file_size
                              and zlib.crc32(target.read_bytes()) == member.CRC)
            if not valid_existing:
                target.write_bytes(z.read(name))  # zipfile validates each member CRC
            downloaded.append(name)
            if len(downloaded) % 25 == 0:
                print('Selected members:', len(downloaded), flush=True)
        save_json(out / 'download_manifest.json', dict(repo=REPO, revision=revision,
                  archive=args.archive, sequences=selected, stride=args.stride,
                  max_frame=args.max_frame, metadata_only=args.metadata_only, members=downloaded))
        print('Saved', len(downloaded), 'members to', out)


def parser(sub):
    p = sub.add_parser('download', help='Inspect/download a small official tracking subset')
    p.add_argument('--archive', default='train.zip', choices=['train.zip', 'test.zip', 'challenge2023.zip'])
    p.add_argument('--revision')
    p.add_argument('--output', default='data/soccernet')
    p.add_argument('--list', action='store_true')
    p.add_argument('--sequences', nargs='+')
    p.add_argument('--metadata-only', action='store_true')
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--max-frame', type=int, default=0)
    p.set_defaults(func=main)
