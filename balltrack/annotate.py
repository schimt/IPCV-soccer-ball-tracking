"""Manual review GUI: optional proposals remain unreviewed until explicitly checked."""
from pathlib import Path
import json
import re
from collections import Counter
from datetime import datetime, timezone

import cv2

from .common import digest, read_rows, reviewed_annotation, row, save_json, video_info, write_rows


class AnnotationSession:
    """Refuse stale saves when a CSV or its schedule changes in another session."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta_path = self.path.with_suffix('.meta.json')
        self.csv_hash = digest(self.path)
        self.meta_text = self.meta_path.read_text(encoding='utf-8')
        self.conflicted = False

    def check(self, rows):
        if self.conflicted:
            raise RuntimeError('Annotation session already stopped after a file conflict')
        if (not self.path.exists() or digest(self.path) != self.csv_hash
                or not self.meta_path.exists()
                or self.meta_path.read_text(encoding='utf-8') != self.meta_text):
            self.conflicted = True
            recovery = self.path.parent / 'backups' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ_session_conflict')
            write_rows(recovery / self.path.name, rows)
            (recovery / self.meta_path.name).write_text(self.meta_text, encoding='utf-8')
            raise RuntimeError('Annotations changed outside this window. Current files were not overwritten. '
                               f'This window\'s copy is saved at {recovery}. Close old annotation windows and restart.')

    def save(self, rows, metadata=None):
        self.check(rows)
        write_rows(self.path, rows)
        self.csv_hash = digest(self.path)
        if metadata is not None:
            save_json(self.meta_path, metadata)
            self.meta_text = self.meta_path.read_text(encoding='utf-8')


def initialize(video, output, stride=1, source='manual'):
    if stride < 1:
        raise ValueError('Positive stride required')
    info = video_info(video)
    path = Path(output)
    meta_path = path.with_suffix('.meta.json')
    if path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['sha256'] != info['sha256'] or meta['annotation_stride'] != stride:
            raise ValueError('Existing annotations belong to another video or stride')
        rows = read_rows(path, 'gt')
        if [r['frame'] for r in rows] != list(range(0, info['frames'], stride)):
            expected = len(range(0, info['frames'], stride))
            raise ValueError(f'Annotation CSV has {len(rows)} rows, but stride {stride} requires '
                             f'{expected} frames from this video. The CSV and metadata do not match; '
                             'close old annotation windows and repair the CSV from backup. Do not change the video.')
        return rows, meta
    info.update(annotation_stride=stride, source=source, interpolation=False,
                tool='balltrack OpenCV manual annotator', index_origin=0)
    from .test_clip import REGISTRY, validate_registered_video
    if REGISTRY.exists() and Path(video).resolve() == Path(json.loads(REGISTRY.read_text())['clip']['video']):
        registered = validate_registered_video(video)
        if stride != registered['annotation_stride']:
            raise ValueError('Use the registered final test annotation stride')
        info.update(role='sole_final_test', source_url=registered['source_url'],
                    source_start_frame=registered['source_start_frame'])
    rows = [row(i, info['fps'], 'unlabeled', source=source) for i in range(0, info['frames'], stride)]
    save_json(meta_path, info)
    write_rows(path, rows)
    return rows, info


def annotation_progress(rows):
    return dict(total=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
                reviewed=sum(reviewed_annotation(r) for r in rows),
                unreviewed=sum(not reviewed_annotation(r) for r in rows),
                sources=dict(Counter(r['source'] for r in rows)))


def first_pending(rows):
    return next((i for i, r in enumerate(rows) if not reviewed_annotation(r)), 0)


def pending_neighbor(rows, index, direction=1, wrap=False):
    """Navigate pending rows without ever displaying an approved annotation."""
    order = (list(range(index+1, len(rows))) if direction == 1
             else list(range(index-1, -1, -1)))
    if wrap:
        order += (list(range(index)) if direction == 1
                  else list(range(len(rows)-1, index, -1)))
    fallback = index if not reviewed_annotation(rows[index]) else None
    return next((i for i in order if not reviewed_annotation(rows[i])), fallback)


def expand_review_rows(current, proposals, frames, fps):
    """Preserve current labels; only restore missing frames as unreviewed proposals."""
    existing = {r['frame']: r for r in current}
    old = {r['frame']: r for r in proposals}
    if len(existing) != len(current) or any(i < 0 or i >= frames for i in existing):
        raise ValueError('Invalid current frame schedule')
    return [dict(existing[i]) if i in existing else
            dict(old[i], reviewed=0) if i in old else
            row(i, fps, 'unlabeled', source='manual') for i in range(frames)]


def review_action(rows, index, key, fps):
    """Return the explicitly reviewed row and next selected index; caller saves first."""
    if key == 'r':
        if rows[index]['status'] == 'unlabeled':
            raise ValueError('Choose a box or visibility status first.')
        value = dict(rows[index], reviewed=1, source='manual_review')
    else:
        status = {'o': 'occluded', 'u': 'uncertain', 'f': 'out_of_frame'}[key]
        value = row(rows[index]['frame'], fps, status, source='manual', reviewed=True)
    return value, min(len(rows)-1, index+1)


def interpolate_between(rows, left, right, fps):
    """Only fill unlabeled interior rows; preserve all visibility decisions and labels."""
    if right <= left + 1 or rows[left]['status'] != 'visible' or rows[right]['status'] != 'visible':
        raise ValueError('Select two visible endpoint boxes with at least one frame between them')
    if any(r['status'] != 'unlabeled' for r in rows[left+1:right]):
        raise ValueError('Interpolation cannot overwrite labels or cross known visibility changes')
    a, b = rows[left], rows[right]
    for i in range(left+1, right):
        fraction = (rows[i]['frame'] - a['frame']) / (b['frame'] - a['frame'])
        box = [float(a[k]) + fraction*(float(b[k])-float(a[k])) for k in ['x', 'y', 'w', 'h']]
        rows[i] = row(rows[i]['frame'], fps, 'visible', box, source='interpolated_linear', reviewed=False)


def check_annotations(args):
    path = Path(args.annotations)
    rows = read_rows(path, 'gt')
    meta = json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))
    actual = video_info(meta['video'])
    if actual['sha256'] != meta['sha256']:
        raise ValueError('Annotation video has changed')
    if meta.get('role') == 'sole_final_test':
        from .test_clip import validate_registered_video
        registered = validate_registered_video(meta['video'])
        if meta['annotation_stride'] != registered['annotation_stride']:
            raise ValueError('Annotation schedule differs from the registered final test')
        if args.require_complete and not all(reviewed_annotation(r, final=True) for r in rows):
            raise ValueError('Final evaluation is blocked: labels remain missing or unreviewed manual proposals')
    if [r['frame'] for r in rows] != list(range(0, actual['frames'], meta['annotation_stride'])):
        raise ValueError('Incomplete or unexpected frame schedule')
    for r in rows:
        if abs(r['timestamp']-r['frame']/actual['fps']) > 1e-6:
            raise ValueError('Timestamp/frame mismatch')
        if r['status'] == 'visible':
            x, y, w, h = (r[k] for k in ['x', 'y', 'w', 'h'])
            if any(v is None for v in [x, y, w, h]) or not (0 <= x < x+w <= actual['width'] and 0 <= y < y+h <= actual['height']):
                raise ValueError(f'Invalid box at frame {r["frame"]}')
    progress = annotation_progress(rows)
    save_json(path.with_suffix('.progress.json'), progress)
    print(json.dumps(progress, indent=2))
    if args.require_complete and progress['unreviewed']:
        raise ValueError('Final evaluation is blocked: labels remain missing or unreviewed')


def import_draft(args):
    """Import visually selected anchors; never silently certify proposals as ground truth."""
    spec = json.loads(Path(args.draft).read_text(encoding='utf-8'))
    path = Path(args.annotations)
    rows = read_rows(path, 'gt')
    meta = json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))
    if spec['video_sha256'] != meta['sha256']:
        raise ValueError('Draft belongs to a different clip')
    if any(r['status'] != 'unlabeled' or r['reviewed'] for r in rows):
        raise ValueError('Draft import requires an untouched annotation file; existing work is preserved')
    index = {r['frame']: i for i, r in enumerate(rows)}
    for interval in spec.get('status_intervals', []):
        if interval['status'] not in {'occluded', 'uncertain', 'out_of_frame'}:
            raise ValueError('Intervals cannot invent visible coordinates')
        for frame in range(interval['first'], interval['last']+1):
            if frame not in index:
                continue
            i = index[frame]
            rows[i] = row(frame, meta['fps'], interval['status'], source='assistant_interval_proposal', reviewed=False)
    for anchor in spec.get('anchors', []):
        if anchor['frame'] not in index:
            continue
        i = index[anchor['frame']]
        rows[i] = row(anchor['frame'], meta['fps'], anchor['status'], anchor.get('box'),
                      source='assistant_visual_keyframe', reviewed=False)
    for left, right in spec.get('interpolate', []):
        if left in index and right in index and index[right] > index[left]+1:
            interpolate_between(rows, index[left], index[right], meta['fps'])
    write_rows(path, rows)
    meta.update(interpolation=bool(spec.get('interpolate')),
                assistance='Assistant visual keyframes and sparse-shot interval proposals; all require human review',
                draft_spec=str(Path(args.draft).resolve()), source='manual_review_with_documented_assistance')
    save_json(path.with_suffix('.meta.json'), meta)
    save_json(path.with_suffix('.progress.json'), annotation_progress(rows))
    with path.with_suffix('.audit.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(action='import_unreviewed_visual_draft', draft=str(args.draft),
                                time_utc=datetime.now(timezone.utc).isoformat()))+'\n')
    print(json.dumps(annotation_progress(rows), indent=2))


def main(args):
    all_rows, info = initialize(args.video, args.output, args.stride)
    frames_file = getattr(args, 'frames_file', None)
    rows = all_rows
    if frames_file:
        text = Path(frames_file).read_text(encoding='utf-8')
        frames = [int(n) for n in re.findall(r'^\|\s*(\d+)\s*\|', text, re.MULTILINE)]
        if not frames or len(frames) != len(set(frames)):
            raise ValueError('Review list must contain unique frame indices in a Markdown table')
        available = {r['frame']: r for r in all_rows}
        if any(f not in available for f in frames):
            raise ValueError('Review list contains a frame outside the annotation schedule')
        rows = [available[f] for f in sorted(frames)]

    def complete_rows():
        selected = {r['frame']: r for r in rows}
        return [selected.get(r['frame'], r) for r in all_rows]
    if args.init_only:
        progress = annotation_progress(rows)
        print(f'Annotation file ready: {len(rows)} frames, {progress["reviewed"]} reviewed, '
              f'{progress["unreviewed"]} pending. Existing annotations are preserved.')
        return
    pending_only = getattr(args, 'pending_only', False)
    if pending_only and not annotation_progress(rows)['unreviewed']:
        save_json(Path(args.output).with_suffix('.progress.json'), annotation_progress(complete_rows()))
        print('All frames have been reviewed; no pending frames to display.')
        return
    session = AnnotationSession(args.output)
    cap = cv2.VideoCapture(args.video)
    index = first_pending(rows)
    scale = min(1., 1280 / info['width'], 800 / info['height'])
    window = 'Ball annotation - manual review (q saves and exits)'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    drag = {'start': None, 'end': None, 'box': None, 'viewport': None}
    header = 108
    view_x, view_y, view_scale = 0, 0, scale
    playing, interpolation_anchor = False, None
    notice = ('Pending frames only. R confirms; approved frames are skipped.' if pending_only
              else 'Inspect each frame. Draft/interpolated boxes require R to review.')
    audit = Path(args.output).with_suffix('.audit.jsonl')

    def commit(i, value, action):
        old = dict(rows[i])
        rows[i] = value
        session.save(complete_rows())
        with audit.open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(), action=action,
                                    frame=value['frame'], before=old, after=value)) + '\n')

    def save_drawn_box():
        if drag['box'] is not None:
            commit(index, row(rows[index]['frame'], info['fps'], 'visible', drag['box'],
                              source='manual', reviewed=False), 'draw_box_pending_review')
            drag['box'] = None

    def mouse(event, x, y, flags, param):
        if y < header:
            return
        px = max(0, min(info['width']-1, view_x + x/view_scale))
        py = max(0, min(info['height']-1, view_y + (y-header)/view_scale))
        if event == cv2.EVENT_RBUTTONDOWN:
            if drag['viewport'] is None:
                vx, vy = max(0, min(info['width']-300, round(px)-150)), max(0, min(info['height']-240, round(py)-120))
                drag['viewport'] = (vx, vy, min(300, info['width']), min(240, info['height']))
            else:
                drag['viewport'] = None
            drag['start'] = None
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.update(start=(px, py), end=(px, py), box=None)
        elif event == cv2.EVENT_MOUSEMOVE and drag['start'] is not None:
            drag['end'] = (px, py)
        elif event == cv2.EVENT_LBUTTONUP and drag['start'] is not None:
            x0, y0 = drag['start']
            box = (min(x0, px), min(y0, py), abs(px - x0), abs(py - y0))
            if box[2] >= 1 and box[3] >= 1:
                drag['box'] = box
            drag['start'] = None

    cv2.setMouseCallback(window, mouse)
    last_index, frame = None, None
    try:
        while True:
            if index != last_index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, rows[index]['frame'])
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Cannot decode annotation frame')
                last_index = index
                drag.update(start=None, end=None, box=None)
            save_drawn_box()
            r = rows[index]
            viewport = drag['viewport'] or (0, 0, info['width'], info['height'])
            view_x, view_y, vw, vh = viewport
            view_scale = min(1280/vw, 700/vh)
            canvas = frame.copy()
            if r['status'] == 'visible':
                x, y, w, h = (float(r[k]) for k in ['x', 'y', 'w', 'h'])
                color = (0, 220, 0) if int(r['reviewed']) else (0, 170, 255)
                cv2.rectangle(canvas, (round(x), round(y)), (round(x+w), round(y+h)), color, 1)
            if drag['start'] is not None:
                a, b = drag['start'], drag['end']
                cv2.rectangle(canvas, tuple(round(v) for v in a), tuple(round(v) for v in b), (0, 255, 0), 1)
            display = cv2.resize(canvas[view_y:view_y+vh, view_x:view_x+vw], None, fx=view_scale, fy=view_scale)
            display = cv2.copyMakeBorder(display, header, 0, 0, 0, cv2.BORDER_CONSTANT)
            progress = annotation_progress(rows)
            lines = [f'Frame {r["frame"]}/{info["frames"]-1} | {r["timestamp"]:.2f}s | {r["status"]} | reviewed {progress["reviewed"]}/{len(rows)}',
                     'Drag: draft box | right click: zoom | R: review+next | O/U/F: status+next',
                     'A/D: prev/next | N: next pending | P: play/pause | B/I: interpolation anchors | Q: save/exit',
                     f'{r["source"]} | {"REVIEWED" if int(r["reviewed"]) else "PENDING"} | {notice}']
            for line, text in enumerate(lines):
                cv2.putText(display, text, (8, 22+25*line), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(max(1, round(1000*info['annotation_stride']/info['fps'])) if playing else 30) & 255
            # Mouse release and keypress can arrive in the same waitKey call.
            save_drawn_box()
            if key == ord('q') or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('x'):
                playing = False
                commit(index, row(rows[index]['frame'], info['fps'], 'unlabeled', source='manual'), 'reset_status')
            if key in map(ord, 'rouf'):
                playing = False
                try:
                    updated, next_index = review_action(rows, index, chr(key), info['fps'])
                    commit(index, updated, 'human_review' if key == ord('r') else 'set_status')
                    if pending_only:
                        next_index = pending_neighbor(rows, index, wrap=True)
                        if next_index is None:
                            print('All frames have been reviewed. Saved; closing annotation window.')
                            break
                    index = next_index
                    notice = 'Saved. Review complete; Q exits.' if not annotation_progress(rows)['unreviewed'] else 'Saved; next selected frame.'
                    drag['viewport'] = None
                except ValueError as exc:
                    notice = str(exc)
            if key == ord('p'):
                playing = not playing
                drag['viewport'] = None
            if key == ord('b') and not frames_file:
                interpolation_anchor = index
                notice = f'Interpolation start frame {rows[index]["frame"]}; I at visible end box.'
            if key == ord('i') and interpolation_anchor is not None:
                try:
                    interpolate_between(rows, interpolation_anchor, index, info['fps'])
                    info['interpolation'] = True
                    session.save(complete_rows(), metadata=info)
                    with audit.open('a', encoding='utf-8') as f:
                        f.write(json.dumps(dict(action='interpolate_unreviewed', left=interpolation_anchor, right=index))+'\n')
                    notice = 'Interpolation is pending review on EVERY generated frame.'
                except ValueError as exc:
                    notice = str(exc)
            if key in [ord('d'), 32]:
                playing = False
                index = pending_neighbor(rows, index) if pending_only else min(len(rows)-1, index+1)
            elif key == ord('a'):
                playing = False
                index = pending_neighbor(rows, index, direction=-1) if pending_only else max(0, index-1)
            elif key == ord('n'):
                playing = False
                order = list(range(index+1, len(rows))) + list(range(index+1))
                index = next((i for i in order if not reviewed_annotation(rows[i])), index)
            if playing:
                next_index = pending_neighbor(rows, index) if pending_only else min(len(rows)-1, index+1)
                if next_index == index or next_index == len(rows)-1:
                    playing = False
                index = next_index
    finally:
        try:
            # Edits are saved as they happen, including mouse release immediately before Q.
            # Never rewrite an old in-memory CSV merely because its window is closing.
            if not session.conflicted:
                session.check(complete_rows())
                save_json(Path(args.output).with_suffix('.progress.json'), annotation_progress(complete_rows()))
        finally:
            cap.release()
            cv2.destroyAllWindows()


def parser(sub):
    p = sub.add_parser('import-annotation-draft')
    p.add_argument('--draft', required=True)
    p.add_argument('--annotations', default='data/test/annotations.csv')
    p.set_defaults(func=import_draft)
    p = sub.add_parser('check-annotations', help='Check video identity, geometry and per-frame review progress')
    p.add_argument('--annotations', default='data/test/annotations.csv')
    p.add_argument('--require-complete', action='store_true')
    p.set_defaults(func=check_annotations)
    p = sub.add_parser('annotate', help='Manually review the selected frame schedule')
    p.add_argument('--video', required=True)
    p.add_argument('--output', default='data/test/annotations.csv')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--pending-only', action='store_true', help='Display only unreviewed frames; skip all approved labels')
    p.add_argument('--frames-file', help='Review only frame indices listed in a Markdown table; preserve the full CSV')
    p.add_argument('--init-only', action='store_true')
    p.set_defaults(func=main)
