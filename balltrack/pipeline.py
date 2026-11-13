from pathlib import Path
import time

import cv2
import numpy as np

from .common import config, environment, row, save_json, video_info, write_rows
from .tracking import BallTracker, CameraMotion, ClassicalDetector


def draw_prediction(frame, prediction, label=''):
    status = prediction['status']
    color = (0, 230, 0) if status == 'observed' else (0, 180, 255)
    if status != 'missing':
        x, y, w, h = (float(prediction[k]) for k in ['x', 'y', 'w', 'h'])
        cv2.rectangle(frame, (round(x), round(y)), (round(x+w), round(y+h)), color, 2)
        cv2.circle(frame, (round(x+w/2), round(y+h/2)), 5, color, 1)
    cv2.putText(frame, f'{label} | frame {prediction["frame"]} | {status}', (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    return frame


def run(args):
    cfg = config(args.config)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty run directory to preserve experiment provenance')
    out.mkdir(parents=True, exist_ok=True)
    info = video_info(args.video)
    detector = ClassicalDetector(cfg['classical'])
    nn_info = None
    if args.method == 'nn':
        from .nn import NeuralDetector
        detector = NeuralDetector(cfg['nn'], args.weights, args.pretrained)
        nn_info = detector.metadata
    tracker = BallTracker(cfg['tracker'], info['width'], info['height'], info['fps'])
    motion = CameraMotion(cfg['motion'])
    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(str(out / 'overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                            info['fps'], (info['width'], info['height']))
    if not writer.isOpened():
        raise RuntimeError('Cannot create MP4 writer')
    rows, elapsed, cuts = [], [], []
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not rows:
                detector.detect(frame)  # untimed warm-up, same image, no tracker state
            begin = time.perf_counter()
            affine, cut = motion.update(frame)
            candidates = detector.detect(frame)
            status, box, confidence = tracker.step(candidates, affine, cut)
            elapsed.append(time.perf_counter() - begin)
            prediction = row(len(rows), info['fps'], status, box, confidence, args.method)
            rows.append(prediction)
            if cut:
                cuts.append(prediction['frame'])
            writer.write(draw_prediction(frame, prediction, args.method))
    finally:
        cap.release()
        writer.release()
    if len(rows) != info['frames']:
        raise ValueError(f'Decoded {len(rows)} frames, expected {info["frames"]}; incomplete run')
    write_rows(out / 'predictions.csv', rows)
    duration = time.perf_counter() - started
    save_json(out / 'run.json', dict(video=info, config=cfg, method=args.method,
              environment=environment(), nn=nn_info, frames=len(rows), camera_cut_frames=cuts,
              timing=dict(pipeline_fps=len(rows)/sum(elapsed), end_to_end_fps=len(rows)/duration,
                          median_frame_ms=float(np.median(elapsed)*1000),
                          pipeline_includes='camera motion, detection, association; CPU transfer synchronizes NN',
                          pipeline_excludes='video decode/encode, model loading, one warm-up',
                          end_to_end_includes='decode, warm-up, pipeline, rendering, encoding, CSV write'),
              config_file=str(Path(args.config).resolve())))
    print('Saved predictions and overlay:', out)


def parser(sub):
    p = sub.add_parser('track')
    p.add_argument('--video', required=True)
    p.add_argument('--method', choices=['classical', 'nn'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--weights')
    p.add_argument('--pretrained', action='store_true', help='Explicitly mark COCO-only baseline')
    p.set_defaults(func=run)
