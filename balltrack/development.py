"""SoccerNet-only candidate caches and validation replay, never final-test tuning."""
import argparse
from copy import deepcopy
from itertools import product
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from .common import config, digest, row, save_json, write_rows, read_rows
from .data import ball_boxes, sequence_metadata
from .evaluate import score_rows
from .tracking import BallTracker, CameraMotion, Candidate, ClassicalDetector


def validate_sequences(root, sequences, frames):
    if frames < 1:
        raise ValueError('Positive contiguous frame count required')
    items = []
    for seq in sequences:
        folder = Path(root) / seq
        meta = sequence_metadata(folder)
        if meta['game'] != '9':
            raise ValueError('This development protocol reserves game 9 for validation')
        images = [folder / meta['imdir'] / f'{i+1:06d}.jpg' for i in range(frames)]
        if not all(p.is_file() for p in images):
            raise ValueError(f'Noncontiguous validation images: {seq}')
        items.append((folder, meta, images))
    return items


def cache(args):
    cv2.setNumThreads(2)
    tracking_hash = digest(Path(__file__).with_name('tracking.py'))
    out = Path(args.output)
    if out.exists():
        raise ValueError('Cache already exists; use a new path')
    items = validate_sequences(args.root, args.sequences, args.frames)
    cfg = config(args.config)
    cfg['nn']['confidence'] = args.confidence
    if args.method == 'nn':
        from .nn import NeuralDetector
        detector = NeuralDetector(cfg['nn'], args.weights)
        model = detector.metadata
    else:
        detector, model = ClassicalDetector(cfg['classical']), None
    records = []
    start = time.perf_counter()
    for folder, meta, images in items:
        gt_boxes = ball_boxes(folder, meta)
        motion = CameraMotion(cfg['motion'])
        frames = []
        for i, path in enumerate(images):
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (meta['height'], meta['width']):
                raise ValueError(f'Invalid image {path}')
            affine, cut = motion.update(image)
            candidates = detector.detect(image)
            boxes = gt_boxes.get(i, [])
            gt = row(i, meta['fps'], 'visible' if len(boxes) == 1 else 'uncertain',
                     boxes[0] if len(boxes) == 1 else None, source='SoccerNet_MOT_import', reviewed=True)
            frames.append(dict(gt=gt, affine=affine.tolist(), cut=cut,
                          candidates=[dict(box=list(d.box), score=d.score) for d in candidates],
                          image=str(path.resolve()), image_sha256=digest(path)))
            if (i+1) % 75 == 0:
                print(f'{args.method}: {meta["name"]} {i+1}/{len(images)}', flush=True)
        records.append(dict(metadata=meta, annotation_sha256=digest(folder / 'gt/gt.txt'), frames=frames))
    save_json(out, dict(role='soccernet_validation', method=args.method, config=cfg,
              model=model, sequences=records, elapsed_seconds=time.perf_counter()-start,
              image_source='official JPEGs; no video re-encoding',
              tracking_code_sha256=tracking_hash))
    print('Saved', out, flush=True)


def load_cache(path):
    data = json.loads(Path(path).read_text())
    if data['role'] != 'soccernet_validation' or any(s['metadata']['game'] != '9' for s in data['sequences']):
        raise ValueError('Only reserved SoccerNet validation data may select parameters')
    return data


def replay(data, cfg, confidence, output=None):
    per_sequence, diagnostics = [], []
    totals = dict(tp=0, fp=0, fn=0)
    for seq in data['sequences']:
        meta = seq['metadata']
        tracker = BallTracker(cfg['tracker'], meta['width'], meta['height'], meta['fps'])
        predictions = []
        oracle = top1 = available_but_missed = 0
        threshold = math.hypot(meta['width'], meta['height']) * .01
        for f in seq['frames']:
            candidates = [Candidate(tuple(d['box']), d['score']) for d in f['candidates'] if d['score'] >= confidence]
            status, box, score = tracker.step(candidates, np.array(f['affine']), f['cut'])
            pred = row(f['gt']['frame'], meta['fps'], status, box, score, data['method'])
            predictions.append(pred)
            gt = f['gt']
            if gt['status'] == 'visible':
                center = np.array([gt['cx'], gt['cy']])
                hit = any(np.linalg.norm(d.center-center) <= threshold for d in candidates)
                oracle += hit
                top1 += bool(candidates) and np.linalg.norm(max(candidates, key=lambda d: d.score).center-center) <= threshold
                correct = box is not None and math.hypot(pred['cx']-gt['cx'], pred['cy']-gt['cy']) <= threshold
                available_but_missed += hit and not correct
        metrics, _, _ = score_rows([f['gt'] for f in seq['frames']], predictions,
                                   meta['width'], meta['height'], meta['fps'])
        metrics['sequence'] = meta['name']
        per_sequence.append(metrics)
        for key in totals:
            totals[key] += metrics[key]
        diagnostics.append(dict(sequence=meta['name'], visible=metrics['visible_frames'],
                           candidate_oracle_hits=int(oracle), top_score_hits=int(top1),
                           candidate_present_but_tracker_missed=int(available_but_missed)))
        if output:
            write_rows(Path(output) / meta['name'] / 'predictions.csv', predictions)
    tp, fp, fn = (totals[k] for k in ['tp', 'fp', 'fn'])
    totals.update(precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0,
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return dict(aggregate=totals, per_sequence=per_sequence, diagnostics=diagnostics)


def diagnose(args):
    data = load_cache(args.cache)
    results = {}
    thresholds = [.001, .01, .03, .05, .15, .4] if data['method'] == 'nn' else [.0]
    for threshold in thresholds:
        results[str(threshold)] = replay(data, data['config'], threshold)
    save_json(args.output, dict(cache=args.cache, cache_sha256=digest(args.cache), thresholds=results))
    panels = []
    for seq in data['sequences']:
        frames = seq['frames']
        for i in [0, len(frames)//2, len(frames)-1]:
            f, meta = frames[i], seq['metadata']
            if f['gt']['status'] != 'visible':
                continue
            frame = cv2.imread(f['image'])
            g = f['gt']
            cx, cy = round(g['cx']), round(g['cy'])
            # Ground truth cyan, all post-NMS candidates red with confidence labels.
            x, y, w, h = (g[k] for k in ['x', 'y', 'w', 'h'])
            cv2.rectangle(frame, (round(x), round(y)), (round(x+w), round(y+h)), (255,255,0), 2)
            for d in f['candidates']:
                x, y, w, h = d['box']
                if abs(x+w/2-cx) < 90 and abs(y+h/2-cy) < 70:
                    cv2.rectangle(frame, (round(x), round(y)), (round(x+w), round(y+h)), (0,0,255), 1)
                    cv2.putText(frame, f'{d["score"]:.3f}', (round(x),round(y)-3), cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,255),1)
            x0, y0 = max(0, min(meta['width']-200,cx-100)), max(0,min(meta['height']-150,cy-75))
            panel = cv2.resize(frame[y0:y0+150,x0:x0+200], (400,300))
            cv2.putText(panel, f'{meta["name"]} frame {i}', (6,22), cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
            panels.append(panel)
    if panels:
        while len(panels)%3:
            panels.append(np.zeros_like(panels[0]))
        cv2.imwrite(str(Path(args.output).with_suffix('.jpg')), np.vstack([np.hstack(panels[i:i+3]) for i in range(0,len(panels),3)]))
    print(json.dumps(results, indent=2))


def select(args):
    out = Path(args.output)
    if out.exists():
        raise ValueError('Use a new selection directory')
    out.mkdir(parents=True)
    trials, best = [], None
    schedule = None
    for path in args.caches:
        data = load_cache(path)
        current = [(s['metadata']['name'], [(f['gt']['frame'],f['image_sha256'],f['gt']) for f in s['frames']])
                   for s in data['sequences']]
        if schedule is not None and current != schedule:
            raise ValueError('Compare the same images and source annotation rows')
        schedule = current
        confidences = [.01,.03,.1,.25] if data['method'] == 'nn' else [.28]
        # Small explicit grid; no parameters are read from the final test.
        for confidence, acquire, gate, gap, margin in product(confidences,
                ([.1,.4] if data['method']=='nn' else [.4,.55]), [.02,.07], [0.,.24],
                ([None] if data['method']=='nn' else [None,.05,.15])):
            cfg = deepcopy(data['config'])
            cfg['tracker'].update(min_acquire_score=max(confidence, acquire),
                                  gate_diagonal_fraction=gate,max_gap_seconds=gap)
            cfg['tracker']['reacquire_score_margin'] = margin
            cfg['nn']['confidence'] = confidence
            result = replay(data, cfg, confidence)
            trial = dict(cache=str(path), confidence=confidence, tracker=cfg['tracker'], **result)
            trials.append(trial)
            # Micro F1, then precision, then recall; stable first choice for exact ties.
            rank = tuple(result['aggregate'][k] for k in ['f1','precision','recall'])
            if best is None or rank > best[0]:
                best = rank, data, cfg, confidence, path, result
        print('Evaluated',path,flush=True)
    _, data, cfg, confidence, path, result = best
    if data['method'] == 'nn':
        cfg['nn']['weights'] = data['model']['weights']
    replay(data, cfg, confidence, out / 'predictions')
    (out / 'selected.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    save_json(out / 'selection.json',dict(selection_metric='micro F1; precision then recall tie-break',
               data_role='SoccerNet game 9 validation only',caches={p:digest(p) for p in args.caches},
               tracking_code_sha256=digest(Path(__file__).with_name('tracking.py')),
               development_code_sha256=digest(__file__),
               selected_cache=str(path),selected_confidence=confidence,config=cfg,results=result,trials=trials))
    print('Selected',path,result['aggregate'],flush=True)


def export(args):
    """Render paired development videos from already selected cached predictions."""
    selection_path = Path(args.selection) / 'selection.json'
    selected = json.loads(selection_path.read_text())
    data = load_cache(selected['selected_cache'])
    baseline = load_cache(args.baseline)
    if data['method'] != baseline['method']:
        raise ValueError('Compare baseline and selected versions of the same method')
    out = Path(args.output)
    if out.exists():
        raise ValueError('Export already exists')
    out.mkdir(parents=True)
    original_cfg = config(args.baseline_config)
    original_conf = original_cfg['nn']['confidence'] if data['method']=='nn' else original_cfg['classical']['min_score']
    baseline_results = replay(baseline,original_cfg,original_conf,out/'baseline_predictions')
    replay(data,selected['config'],selected['selected_confidence'],out/'selected_predictions')
    panels=[]
    for seq in data['sequences']:
        meta=seq['metadata']; name=meta['name']
        a=read_rows(out/'baseline_predictions'/name/'predictions.csv','pred')
        b=read_rows(out/'selected_predictions'/name/'predictions.csv','pred')
        writer=cv2.VideoWriter(str(out/(name+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),meta['fps'],(960,540))
        if not writer.isOpened():
            raise ValueError('Cannot create development video')
        seen=set()
        try:
            for f,pa,pb in zip(seq['frames'],a,b):
                frame=cv2.imread(f['image']); gt=f['gt']
                for r,color in [(gt,(255,255,0)),(pa,(0,180,255)),(pb,(0,255,0))]:
                    if r['status'] not in {'visible','observed','predicted'}:
                        continue
                    x,y,w,h=(float(r[k]) for k in ['x','y','w','h'])
                    cv2.rectangle(frame,(round(x),round(y)),(round(x+w),round(y+h)),color,2)
                    cv2.circle(frame,(round(x+w/2),round(y+h/2)),5,color,2)
                cv2.putText(frame,f'DEVELOPMENT game 9 | {name} f{gt["frame"]} | {data["method"]}',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                cv2.putText(frame,'cyan: source GT | orange: baseline | green: selected',(20,70),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                cv2.putText(frame,f'baseline {pa["status"]} | selected {pb["status"]} | GT {gt["status"]}',(20,105),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                writer.write(cv2.resize(frame,(960,540)))
                if gt['status']=='visible':
                    correct=pb['cx'] is not None and math.hypot(pb['cx']-gt['cx'],pb['cy']-gt['cy']) <= math.hypot(meta['width'],meta['height'])*.01
                    category='success' if correct else 'failure'
                    if category not in seen:
                        seen.add(category)
                        cv2.imwrite(str(out/f'{name}_{category}_{gt["frame"]:04d}.jpg'),frame)
                        panels.append(cv2.resize(frame,(640,360)))
        finally:
            writer.release()
    if panels:
        if len(panels)%2:
            panels.append(np.zeros_like(panels[0]))
        cv2.imwrite(str(out/'examples.jpg'),np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)]))
    save_json(out/'export.json',dict(role='SoccerNet development only',selection_sha256=digest(selection_path),
              selected_cache_sha256=digest(selected['selected_cache']),baseline_cache_sha256=digest(args.baseline),
              baseline=baseline_results,selected=selected['results'],
              example_rule='First selected success and first selected failure on each sequence; eligible frames only'))
    print('Exported paired development videos:',out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    c = sub.add_parser('cache')
    c.add_argument('--root', required=True)
    c.add_argument('--sequences', nargs='+', required=True)
    c.add_argument('--frames', type=int, default=300)
    c.add_argument('--config', default='configs/default.yaml')
    c.add_argument('--method', choices=['classical','nn'], required=True)
    c.add_argument('--weights')
    c.add_argument('--confidence', type=float, default=.001)
    c.add_argument('--output', required=True)
    c.set_defaults(func=cache)
    d = sub.add_parser('diagnose')
    d.add_argument('--cache', required=True)
    d.add_argument('--output', required=True)
    d.set_defaults(func=diagnose)
    s = sub.add_parser('select')
    s.add_argument('--caches',nargs='+',required=True)
    s.add_argument('--output',required=True)
    s.set_defaults(func=select)
    e=sub.add_parser('export')
    e.add_argument('--selection',required=True)
    e.add_argument('--baseline',required=True)
    e.add_argument('--baseline-config',default='configs/default.yaml')
    e.add_argument('--output',required=True)
    e.set_defaults(func=export)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
