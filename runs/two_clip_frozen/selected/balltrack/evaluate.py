"""Single-ball localization metrics, explicit missing/invisible handling, paired evaluation."""
from pathlib import Path
import csv
import json
import math

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import config, read_rows, reviewed_annotation, save_json
from .pipeline import draw_prediction


def score_rows(gt, predictions, width, height, fps, threshold_fraction=.01, stride=1, observed_only=False):
    predictions = {r['frame']: r for r in predictions}
    diagonal = math.hypot(width, height)
    threshold = diagonal * threshold_fraction
    tp = fp = fn = visible = ignored = invisible_predictions = 0
    errors, penalties, details, losses = [], [], [], []
    loss_start, loss_count = None, 0
    previous_frame = None

    def end_loss():
        nonlocal loss_start, loss_count
        if loss_count:
            losses.append(dict(start_frame=loss_start, annotated_frames=loss_count,
                               duration_seconds=loss_count * stride / fps))
        loss_start, loss_count = None, 0

    for g in gt:
        if not reviewed_annotation(g):
            raise ValueError('All evaluated GT frames must be manually reviewed and labeled')
        p = predictions.get(g['frame'])
        if previous_frame is not None and g['frame'] != previous_frame + stride:
            end_loss()
        previous_frame = g['frame']
        accepted = p is not None and p['status'] in ({'observed'} if observed_only else {'observed', 'predicted'})
        correct, error = False, None
        if g['status'] == 'visible':
            visible += 1
            if accepted:
                error = math.hypot(p['cx'] - g['cx'], p['cy'] - g['cy'])
                errors.append(error)
                correct = error <= threshold
            tp += int(correct)
            fn += int(not correct)
            fp += int(accepted and not correct)
            penalties.append(min(error / diagonal, 1.) if error is not None else 1.)
            if correct:
                end_loss()
            else:
                loss_start = g['frame'] if loss_start is None else loss_start
                loss_count += 1
        elif g['status'] == 'uncertain':
            ignored += 1
            end_loss()
        else:  # occluded/out_of_frame: never invent an invisible reference center
            fp += int(p is not None and p['status'] == 'observed')
            invisible_predictions += int(p is not None and p['status'] == 'predicted')
            end_loss()
        details.append(dict(frame=g['frame'], timestamp=g['timestamp'], gt_status=g['status'],
                            prediction_status=p['status'] if p else 'missing',
                            correct=int(correct), center_error_px=error))
    end_loss()
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / visible if visible else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    metrics = dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1,
                   visible_frames=visible, uncertain_frames_excluded=ignored,
                   visible_localization_coverage=recall, threshold_px=threshold,
                   threshold_diagonal_fraction=threshold_fraction,
                   mean_center_error_px=float(np.mean(errors)) if errors else None,
                   median_center_error_px=float(np.median(errors)) if errors else None,
                   mean_center_error_normalized=float(np.mean(errors)/diagonal) if errors else None,
                   visible_frames_with_position=len(errors),
                   mean_capped_normalized_error_missing_penalty_1=float(np.mean(penalties)) if penalties else None,
                   lost_periods=len(losses), longest_loss_seconds=max((r['duration_seconds'] for r in losses), default=0),
                   invisible_predicted_positions_not_scored=invisible_predictions)
    return metrics, details, losses


def main(args):
    cfg = config(args.config)
    gt_path = Path(args.gt)
    meta = json.loads(gt_path.with_suffix('.meta.json').read_text(encoding='utf-8'))
    gt = read_rows(gt_path, 'gt')
    final_test = args.require_test_minute or meta.get('role') == 'sole_final_test'
    stride = meta['annotation_stride']
    expected = list(range(0, meta['frames'], stride))
    if [r['frame'] for r in gt] != expected:
        raise ValueError('GT must cover the full declared frame schedule')
    if final_test:
        from .test_clip import validate_registered_video
        registered = validate_registered_video(meta['video'])
        if (meta['sha256'] != registered['clip']['sha256']
                or abs(meta['duration_seconds']-60) > 1e-6
                or meta['frames'] != registered['clip']['frames']
                or meta['fps'] != registered['clip']['fps']
                or stride != registered['annotation_stride']):
            raise ValueError('Final evaluation requires the registered 60-second clip and annotation schedule')
        if len(args.predictions) != 2:
            raise ValueError('Final evaluation must compare both methods together on the sole test clip')
    for g in gt:
        if not reviewed_annotation(g, final=final_test):
            raise ValueError('All evaluated GT frames must be manually reviewed and labeled')
        if abs(g['timestamp'] - g['frame']/meta['fps']) > 1e-4:
            raise ValueError('GT timestamp/frame mismatch')
        if g['status'] == 'visible' and not (0 <= g['cx'] < meta['width'] and 0 <= g['cy'] < meta['height']):
            raise ValueError('GT center outside image')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary, all_details, methods = [], {}, []
    fig, ax = plt.subplots(figsize=(10, 4))
    for spec in args.predictions:
        name, path = spec.split('=', 1)
        if name in all_details or not name.replace('_', '').isalnum():
            raise ValueError('Use unique alphanumeric method names')
        predictions = read_rows(path, 'pred')
        run = json.loads(Path(path).with_name('run.json').read_text(encoding='utf-8'))
        methods.append(run['method'])
        if final_test and (run['method'] not in {'classical', 'nn'} or len(set(methods)) != len(methods)):
            raise ValueError('Final comparison requires one classical and one NN run')
        if run['video']['sha256'] != meta['sha256']:
            raise ValueError('Predictions and GT belong to different videos')
        if [r['frame'] for r in predictions] != list(range(meta['frames'])):
            raise ValueError('Prediction CSV must include every video frame, including explicit missing rows')
        if any(abs(r['timestamp']-r['frame']/meta['fps']) > 1e-4 for r in predictions):
            raise ValueError('Prediction timestamp/frame mismatch')
        for observed_only in [False, True]:
            metrics, details, losses = score_rows(gt, predictions, meta['width'], meta['height'], meta['fps'],
                          cfg['evaluation']['distance_diagonal_fraction'], stride, observed_only)
            metrics.update(method=name, mode='observed_only' if observed_only else 'tracking',
                           pipeline_fps=run['timing']['pipeline_fps'], end_to_end_fps=run['timing']['end_to_end_fps'])
            summary.append(metrics)
            if not observed_only:
                all_details[name] = details
                save_json(out / f'{name}_lost_periods.json', losses)
                with (out / f'{name}_frames.csv').open('w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, details[0].keys()); w.writeheader(); w.writerows(details)
                ax.plot([d['timestamp'] for d in details],
                        [d['center_error_px'] if d['center_error_px'] is not None else np.nan for d in details], label=name, alpha=.7)
                # Representative success/failure frames, selected by a fixed rule, not cherry-picked.
                cap = cv2.VideoCapture(meta['video'])
                by_frame = {p['frame']: p for p in predictions}
                for category, correct in [('success', 1), ('failure', 0)]:
                    examples = [d for d in details if d['gt_status'] == 'visible' and d['correct'] == correct][:3]
                    for d in examples:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, d['frame']); ok, image = cap.read()
                        if not ok:
                            continue
                        draw_prediction(image, by_frame[d['frame']], name)
                        g = next(g for g in gt if g['frame'] == d['frame'])
                        cv2.circle(image, (round(g['cx']), round(g['cy'])), 8, (255, 255, 0), 2)
                        cv2.imwrite(str(out / f'{name}_{category}_{d["frame"]:06d}.jpg'), image)
                cap.release()
    ax.axhline(summary[0]['threshold_px'], color='black', linestyle='--', label='correctness threshold')
    ax.set(xlabel='Time (s)', ylabel='Center error (pixels)', title='Errors for returned positions; missing positions are gaps')
    ax.legend(); fig.tight_layout(); fig.savefig(out / 'center_errors.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3))
    for i, (name, details) in enumerate(all_details.items()):
        visible = [d for d in details if d['gt_status'] == 'visible']
        ax.scatter([d['timestamp'] for d in visible], [i]*len(visible),
                   c=['green' if d['correct'] else 'red' for d in visible], s=8, marker='|')
    ax.set_yticks(range(len(all_details)), list(all_details)); ax.set_xlabel('Time (s)')
    ax.set_title('Reviewed visible samples: green = localized; red = missing or incorrect')
    fig.tight_layout(); fig.savefig(out / 'coverage.png'); plt.close(fig)
    save_json(out / 'metrics.json', dict(dataset_source=meta['source'], video_sha256=meta['sha256'],
                annotation_stride=stride, annotation_frames=expected, annotation_count=len(gt),
                video_frames=meta['frames'], annotation_hz=meta['fps']/stride,
                loss_duration_is_sampled_approximation=stride > 1, metrics=summary))
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, summary[0].keys()); w.writeheader(); w.writerows(summary)
    print('Saved paired evaluation:', out / 'metrics.csv')


def parser(sub):
    p = sub.add_parser('evaluate')
    p.add_argument('--gt', required=True)
    p.add_argument('--predictions', nargs='+', required=True, help='name=path/predictions.csv')
    p.add_argument('--output', default='results/test')
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--require-test-minute', action='store_true')
    p.set_defaults(func=main)
