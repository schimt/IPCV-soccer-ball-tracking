from pathlib import Path
import csv
import hashlib
import json
import math
import platform

import cv2
import yaml

FIELDS = ['frame', 'timestamp', 'status', 'cx', 'cy', 'x', 'y', 'w', 'h',
          'confidence', 'source', 'reviewed']
GT_STATUSES = {'visible', 'occluded', 'uncertain', 'out_of_frame', 'unlabeled'}
PRED_STATUSES = {'observed', 'predicted', 'missing'}


def reviewed_annotation(r, final=False):
    """Proposals cannot become GT merely by carrying a review flag."""
    source = r.get('source', '')
    return (int(r.get('reviewed', 0)) == 1 and r['status'] != 'unlabeled'
            and not source.startswith(('interpolated', 'assistant_'))
            and (not final or source in {'manual', 'manual_review'}))


def config(path='configs/default.yaml'):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or n <= 0:
        raise ValueError('Video must have a known, constant frame rate and frame count.')
    return dict(video=str(Path(path).resolve()), sha256=digest(path), fps=fps,
                frames=n, width=width, height=height, duration_seconds=n / fps)


def row(frame, fps, status, box=None, confidence=None, source='', reviewed=False):
    r = dict.fromkeys(FIELDS, '')
    r.update(frame=int(frame), timestamp=frame / fps, status=status,
             source=source, reviewed=int(reviewed))
    if box is not None:
        x, y, w, h = map(float, box)
        r.update(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2)
    if confidence is not None:
        r['confidence'] = float(confidence)
    return r


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_rows(path, kind):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for r in rows:
        r['frame'] = int(r['frame'])
        if r['frame'] < 0 or r['frame'] in seen:
            raise ValueError('Negative or duplicate frame index')
        seen.add(r['frame'])
        r['timestamp'] = float(r['timestamp'])
        if not math.isfinite(r['timestamp']) or r['timestamp'] < 0:
            raise ValueError('Invalid timestamp')
        for k in ['cx', 'cy', 'x', 'y', 'w', 'h', 'confidence']:
            r[k] = float(r[k]) if r.get(k, '') != '' else None
            if r[k] is not None and not math.isfinite(r[k]):
                raise ValueError(f'Non-finite {k}')
        r['reviewed'] = int(r.get('reviewed', 0) or 0)
        allowed = GT_STATUSES if kind == 'gt' else PRED_STATUSES
        if r['status'] not in allowed:
            raise ValueError(f'Invalid {kind} status: {r["status"]}')
        positioned = r['status'] in {'visible', 'observed', 'predicted'}
        if positioned != (r['cx'] is not None and r['cy'] is not None):
            raise ValueError('Position must be present exactly for localized rows')
        if not positioned and any(r[k] is not None for k in ['cx', 'cy', 'x', 'y', 'w', 'h']):
            raise ValueError('Invisible/missing rows cannot contain guessed geometry')
        if positioned and r['w'] is not None and (r['w'] <= 0 or r['h'] is None or r['h'] <= 0):
            raise ValueError('Invalid bounding box')
    return sorted(rows, key=lambda r: r['frame'])


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor(), opencv=cv2.__version__)
