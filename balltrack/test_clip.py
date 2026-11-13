"""Prepare and register the project's single, immutable 60-second final test clip."""
from pathlib import Path
import json
import math

import cv2

from .common import digest, save_json, video_info

PROJECT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT / 'configs/final_test.json'


def extract_interval(source, destination, start_seconds, duration=60):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('Refusing to replace a clip that may already have annotations')
    info = video_info(source)
    fps = info['fps']
    start_float, count_float = start_seconds * fps, duration * fps
    if (start_seconds < 0 or duration <= 0 or
            not math.isclose(start_float, round(start_float), abs_tol=1e-6) or
            not math.isclose(count_float, round(count_float), abs_tol=1e-6)):
        raise ValueError('Start/duration must align with source frames; transcode nonintegral frame rates first')
    start, count = round(start_float), round(count_float)
    if start + count > info['frames']:
        raise ValueError('Requested interval exceeds source duration')
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.stem + '.partial.mp4')
    if pending.exists():
        raise ValueError(f'Previous partial extraction exists: {pending}; inspect it before retrying')
    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(pending), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                            (info['width'], info['height']))
    if not writer.isOpened():
        cap.release()
        raise ValueError('Could not open MP4 encoder')
    timestamps = []
    # Decode sequentially instead of relying on a potentially keyframe-rounded seek.
    try:
        for index in range(start + count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Source stopped decoding at frame {index}')
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if abs(timestamp - index/fps) > 0.5/fps:
                raise ValueError('Source timestamps are not a constant-frame-rate timeline')
            if index >= start:
                writer.write(frame)
                timestamps.append(timestamp)
    finally:
        cap.release()
        writer.release()
    check = video_info(pending)
    verify = cv2.VideoCapture(str(pending))
    decoded, last_time = 0, None
    while True:
        ok, _ = verify.read()
        if not ok:
            break
        last_time = verify.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if abs(last_time - decoded/fps) > 0.5/fps:
            verify.release()
            raise ValueError('Output timestamps are inconsistent')
        decoded += 1
    verify.release()
    if decoded != count or check['frames'] != count or abs(check['duration_seconds'] - duration) > 1e-6:
        raise ValueError('Decoded frame count/duration mismatch in extracted clip')
    pending.rename(destination)
    check['video'] = str(destination.resolve())
    return dict(source=info, clip=check, source_start_frame=start,
                source_end_frame_exclusive=start+count, source_start_seconds=start/fps,
                source_end_seconds_exclusive=(start+count)/fps, source_timestamps_verified=True,
                decoded_output_frames=decoded, last_frame_timestamp=last_time,
                encoding='OpenCV/FFmpeg MPEG-4 Part 2 (mp4v), original pixel dimensions, no audio',
                temporal_resampling=False, spatial_resizing=False)


def validate_registered_video(path):
    registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
    if digest(path) != registry['clip']['sha256']:
        raise ValueError('This is not the registered sole final test clip')
    return registry


def prepare(args):
    source = Path(args.source).resolve()
    destination = PROJECT / 'data/test/clip.mp4'
    if REGISTRY.exists():
        registered = validate_registered_video(destination)
        if (digest(source) != registered['source']['sha256'] or
                args.start != registered['source_start_seconds']):
            raise ValueError('A different source/interval is already registered; no second final test is allowed')
        print('Existing sole final test verified:', destination)
        return
    snapshot = dict(config_sha256=digest(PROJECT / 'configs/default.yaml'),
                    config_text=(PROJECT / 'configs/default.yaml').read_text(encoding='utf-8'),
                    purpose='Development state recorded before clip selection/annotation; no test-driven tuning',
                    pilot_weights_sha256=digest(PROJECT / 'runs/nn_pilot/fit/weights/best.pt')
                    if (PROJECT / 'runs/nn_pilot/fit/weights/best.pt').exists() else None)
    result = extract_interval(source, destination, args.start)
    result.update(role='sole_final_test', source_url=args.url,
                  source_url_provenance='User-supplied; local file is authoritative',
                  selection_reason=args.reason,
                  selection_without_model_predictions=True,
                  annotations=str((PROJECT / 'data/test/annotations.csv').resolve()),
                  annotation_stride=1, permitted_uses=['manual_annotation', 'final_evaluation', 'presentation'],
                  forbidden_uses=['training', 'validation', 'hyperparameter_selection'],
                  development_snapshot=snapshot)
    save_json(REGISTRY, result)
    save_json(PROJECT / 'data/test/clip_manifest.json', result)
    print('Registered sole final test:', destination)
    print('Interval:', result['source_start_seconds'], 'to', result['source_end_seconds_exclusive'], 'seconds')
    print('Verified:', result['decoded_output_frames'], 'frames,', result['clip']['duration_seconds'], 'seconds')


def parser(sub):
    p = sub.add_parser('prepare-test', help='Register exactly one 60-second final test from the chosen local source')
    p.add_argument('--source', default='data/test/source_video.mp4')
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--url', default='https://www.youtube.com/watch?v=t0GESlaVNdE')
    p.add_argument('--reason', default='First contiguous 60 seconds, fixed before inference; retains all shots and visibility changes.')
    p.set_defaults(func=prepare)
