"""SoccerNet MOT-to-YOLO conversion with game-level splitting and preview boxes."""
import configparser
import csv
from pathlib import Path
import random
import re
import shutil

import cv2
import numpy as np
import yaml

from .common import digest, row, save_json, video_info, write_rows


def sequence_metadata(folder):
    folder = Path(folder)
    seq, game = configparser.ConfigParser(), configparser.ConfigParser()
    if not seq.read(folder / 'seqinfo.ini') or not game.read(folder / 'gameinfo.ini'):
        raise ValueError(f'Missing metadata in {folder}')
    s, g = seq['Sequence'], game['Sequence']
    ids = {int(k.split('_')[1]) for k, v in g.items()
           if re.fullmatch(r'trackletid_\d+', k) and v.split(';')[0].strip().lower() == 'ball'}
    if not ids:
        raise ValueError(f'No documented ball tracklet in {folder}; refusing to guess')
    return dict(name=s['name'], game=g['gameID'], ball_ids=sorted(ids),
                width=int(s['imWidth']), height=int(s['imHeight']),
                fps=float(s['frameRate']), frames=int(s['seqLength']), imdir=s['imDir'])


def ball_boxes(folder, metadata, origin=1):
    """MOT uses 1-based frame IDs and top-left coordinates; convert to zero-based pixels.

    Last three columns are placeholders (-1), NOT class or visibility fields.
    """
    boxes = {}
    with (Path(folder) / 'gt/gt.txt').open(newline='') as f:
        for raw in csv.reader(f):
            if len(raw) != 10:
                raise ValueError('Expected the documented 10-column SoccerNet MOT format')
            values = list(map(float, raw))
            frame, track = map(int, values[:2])
            if track not in metadata['ball_ids'] or values[6] <= 0:
                continue
            if not 1 <= frame <= metadata['frames']:
                raise ValueError('MOT frame outside sequence')
            x, y, w, h = values[2:6]
            x, y = x - origin, y - origin
            x2, y2 = min(metadata['width'], x + w), min(metadata['height'], y + h)
            x, y = max(0., x), max(0., y)
            if x2 <= x or y2 <= y:
                continue
            boxes.setdefault(frame - 1, []).append((x, y, x2 - x, y2 - y))
    return boxes


def inspect(args):
    records = []
    for p in sorted(Path(args.root).rglob('gameinfo.ini')):
        meta = sequence_metadata(p.parent)
        boxes = ball_boxes(p.parent, meta, args.mot_origin)
        records.append(dict(**meta, labeled_ball_frames=len(boxes),
                            available_images=len(list((p.parent / meta['imdir']).glob('*')))))
    if not records:
        raise ValueError('No sequences found')
    save_json(args.output, records)
    for r in records:
        print(r)


def convert(args):
    if args.stride < 1:
        raise ValueError('stride must be positive')
    root, out = Path(args.root), Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory to prevent stale split leakage')
    sequences = [(p.parent, sequence_metadata(p.parent)) for p in sorted(root.rglob('gameinfo.ini'))
                 if (p.parent / 'gt/gt.txt').exists()]
    sequences = [(p, m) for p, m in sequences if any((p / m['imdir']).glob('*.jpg'))]
    games = sorted({m['game'] for _, m in sequences})
    if len(games) < 2:
        raise ValueError('At least two different games with images are required')
    val_games = set(args.val_games or [random.Random(args.seed).choice(games)])
    if not val_games.issubset(games) or val_games == set(games):
        raise ValueError('Choose validation games present in the data, leaving training games')
    counts = {'train': 0, 'val': 0}
    manifest, previews, ball_sizes = [], [], []
    for folder, meta in sequences:
        split = 'val' if meta['game'] in val_games else 'train'
        boxes = ball_boxes(folder, meta, args.mot_origin)
        image_out, label_out = out / 'images' / split, out / 'labels' / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        local_count, selected_frames = 0, []
        for image_path in sorted((folder / meta['imdir']).glob('*.jpg')):
            frame = int(image_path.stem) - 1
            if frame % args.stride:
                continue
            frame_boxes = boxes.get(frame, [])
            if not frame_boxes and not args.include_unlabeled_negatives:
                # MOT does not encode per-frame invisibility; missing ball is not a verified negative.
                continue
            img = cv2.imread(str(image_path))
            if img is None or img.shape[:2] != (meta['height'], meta['width']):
                raise ValueError(f'Image dimensions differ from metadata: {image_path}')
            stem = f'{meta["name"]}_{frame:06d}'
            shutil.copy2(image_path, image_out / (stem + '.jpg'))
            lines = []
            for x, y, w, h in frame_boxes:
                lines.append(f'0 {(x+w/2)/meta["width"]:.8f} {(y+h/2)/meta["height"]:.8f} {w/meta["width"]:.8f} {h/meta["height"]:.8f}')
                ball_sizes.append([w, h])
                cv2.rectangle(img, (round(x), round(y)), (round(x+w), round(y+h)), (0, 255, 255), 2)
            (label_out / (stem + '.txt')).write_text('\n'.join(lines), encoding='utf-8')
            if local_count < 3:
                cv2.putText(img, f'{meta["name"]} / game {meta["game"]} / {split} / frame {frame}',
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                previews.append(cv2.resize(img, (640, 360)))
            counts[split] += 1
            local_count += 1
            selected_frames.append(frame)
        manifest.append(dict(**meta, split=split, converted_frames=local_count,
                             frame_indices=selected_frames,
                             annotation_sha256=digest(folder / 'gt/gt.txt'),
                             metadata_sha256=digest(folder / 'gameinfo.ini')))
    if min(counts.values()) == 0:
        raise ValueError('Empty train/validation split after conversion')
    (out / 'dataset.yaml').write_text(yaml.safe_dump(dict(path=out.resolve().as_posix(),
             train='images/train', val='images/val', names={0: 'ball'})), encoding='utf-8')
    save_json(out / 'manifest.json', dict(source=str(root.resolve()), seed=args.seed,
              mot_origin=args.mot_origin, split_unit='gameID', validation_games=sorted(val_games),
              sequences=manifest, counts=counts, include_unlabeled_negatives=args.include_unlabeled_negatives,
              median_ball_wh=np.median(ball_sizes, axis=0).tolist() if ball_sizes else None))
    if previews:
        cv2.imwrite(str(out / 'box_preview.jpg'), np.vstack(previews))
    print('Converted:', counts, '| visually inspect', out / 'box_preview.jpg')


def parser(sub):
    p = sub.add_parser('sequence-video', help='Make a development clip from contiguous SoccerNet frames')
    p.add_argument('--sequence', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--frames', type=int, default=75)
    p.add_argument('--mot-origin', type=int, choices=[0, 1], default=1)
    p.set_defaults(func=sequence_video)
    p = sub.add_parser('inspect-data')
    p.add_argument('--root', default='data/soccernet/train')
    p.add_argument('--output', default='results/data_inventory.json')
    p.add_argument('--mot-origin', type=int, choices=[0, 1], default=1)
    p.set_defaults(func=inspect)
    p = sub.add_parser('convert')
    p.add_argument('--root', default='data/soccernet/train')
    p.add_argument('--output', default='data/yolo')
    p.add_argument('--val-games', nargs='+')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--mot-origin', type=int, choices=[0, 1], default=1)
    p.add_argument('--include-unlabeled-negatives', action='store_true', help='Only after manual negative-frame review')
    p.set_defaults(func=convert)


def sequence_video(args):
    folder, out = Path(args.sequence), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = sequence_metadata(folder)
    boxes = ball_boxes(folder, meta, args.mot_origin)
    images = {int(p.stem)-1: p for p in (folder / meta['imdir']).glob('*.jpg')}
    if args.frames < 1 or not all(i in images for i in range(args.frames)):
        raise ValueError('Contiguous images starting at frame 1 required; download with --stride 1')
    writer = cv2.VideoWriter(str(out / 'sequence.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                            meta['fps'], (meta['width'], meta['height']))
    if not writer.isOpened():
        raise ValueError('Cannot open video writer')
    rows = []
    for i in range(args.frames):
        img = cv2.imread(str(images[i]))
        if img is None:
            raise ValueError(f'Cannot decode {images[i]}')
        writer.write(img)
        b = boxes.get(i, [])
        # MOT presence is used for development localization, NOT claimed manual visibility GT.
        rows.append(row(i, meta['fps'], 'visible' if len(b) == 1 else 'uncertain',
                        b[0] if len(b) == 1 else None, source='SoccerNet_MOT_import', reviewed=True))
    writer.release()
    info = video_info(out / 'sequence.mp4')
    info.update(source='SOCCERNET DEVELOPMENT ONLY (MOT visibility not verified)',
                annotation_stride=1, tool='MOT import', interpolation='unknown in upstream annotation',
                sequence=meta, index_origin=0)
    write_rows(out / 'annotations.csv', rows)
    save_json(out / 'annotations.meta.json', info)
    print('Development video and imported annotations:', out)
