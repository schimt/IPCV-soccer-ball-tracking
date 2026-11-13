"""Reproduce the two-clip results; added after the evaluation for reproduction only.

The original evaluation was run once with the hash-locked `python -m balltrack.two_clip evaluate`,
which also verifies the provenance of artefacts that are not part of this repository.

score  Recompute every metric from the stored prediction CSVs and annotations, without videos,
       using the frozen evaluator, and optionally compare with the stored comparison tables.
track  Rerun the four frozen versions on both clips (requires both videos and retrained weights).
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from .common import digest, read_rows
from .metric_audit import write_csv
from .two_clip import FROZEN, METHODS, REGISTRY, ROOT, VERSIONS, frozen_module, read, require

ANNOTATIONS = {'test_1': Path('data/test/annotations.csv'), 'test_2': Path('data/test_2/annotations.csv')}
TIMING_SOURCE = {'test_1': 'historical', 'test_2': 'new sequential run; OpenCV threads=2'}


def stored_source(clip, version, method):
    if clip == 'test_1':
        return (Path('runs/final_selected_v2') if version == 'selected' else Path('runs/final'))/method
    return Path('runs/two_clip_experiment/test_2')/version/method


def score(args):
    evaluator = frozen_module('evaluate')
    metrics = []
    for clip in read(REGISTRY)['clips']:
        info, gt = clip['video'], read_rows(ANNOTATIONS[clip['id']], 'gt')
        require(len(gt) == info['frames'], f'Incomplete annotations for {clip["id"]}')
        for version in VERSIONS:
            for method in METHODS:
                source = (Path(args.predictions_root)/clip['id']/version/method if args.predictions_root
                          else stored_source(clip['id'], version, method))
                pred, run = read_rows(source/'predictions.csv', 'pred'), read(source/'run.json')
                require([p['frame'] for p in pred] == list(range(info['frames'])), f'Predictions must include every frame: {source}')
                for observed_only in [False, True]:
                    m, _, _ = evaluator.score_rows(gt, pred, info['width'], info['height'], info['fps'], .01, 1, observed_only)
                    m.update(method=method, version=version, test_clip=clip['id'],
                             mode='observed_only' if observed_only else 'tracking', status='evaluated',
                             visible_frames_without_position=m['visible_frames']-m['visible_frames_with_position'],
                             missing_predictions_all_frames=sum(p['status'] == 'missing' for p in pred),
                             pipeline_fps=run['timing']['pipeline_fps'], end_to_end_fps=run['timing']['end_to_end_fps'],
                             timing_source=TIMING_SOURCE[clip['id']],
                             predictions=str(source/'predictions.csv'), predictions_sha256=digest(source/'predictions.csv'))
                    metrics.append(m)
    out = Path(args.output)
    write_csv(out/'comparison_all_modes.csv', metrics)
    write_csv(out/'comparison.csv', [m for m in metrics if m['mode'] == 'tracking'])
    print('Saved', out/'comparison.csv', 'and', out/'comparison_all_modes.csv')
    if args.check:
        ok = all(compare(out/name, ROOT/'completed'/name) for name in ['comparison.csv', 'comparison_all_modes.csv'])
        sys.exit(0 if ok else 1)


def compare(new, stored):
    a, b = (list(csv.DictReader(open(p, newline='', encoding='utf-8'))) for p in (new, stored))
    differences = [(i, k, b[i].get(k), row.get(k)) for i, row in enumerate(a) for k in b[0] if row.get(k) != b[i].get(k)]
    same = len(a) == len(b) and list(a[0]) == list(b[0]) and not differences
    print(f'{stored}: {len(b)} rows x {len(b[0])} columns,', 'identical' if same else f'{len(differences)} differences')
    for i, key, old, value in differences[:20]:
        print(f'  row {i} {key}: stored {old} reproduced {value}')
    return same


def track(args):
    weights = {'selected': args.selected_weights, 'original': args.original_weights}
    for clip in read(REGISTRY)['clips']:
        video = Path(clip['video']['video'])
        require(video.is_file() and digest(video) == clip['video']['sha256'], f'Place the registered video at {video}')
        for version in VERSIONS:
            for method in METHODS:
                require(method == 'classical' or Path(weights[version] or '').is_file(), f'Supply --{version}-weights')
                run_args = dict(video=str(video), method=method, config=str(FROZEN/version/f'{method}.yaml'),
                                output=str(Path(args.output)/clip['id']/version/method),
                                weights=weights[version] if method == 'nn' else None, pretrained=False)
                code = ('import sys,json,cv2; from types import SimpleNamespace; '
                        'sys.path.insert(0,sys.argv[1]); from balltrack.pipeline import run; '
                        'cv2.setNumThreads(2); run(SimpleNamespace(**json.loads(sys.argv[2])))')
                subprocess.run([sys.executable, '-B', '-c', code, str((FROZEN/version).resolve()), json.dumps(run_args)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('score')
    s.add_argument('--predictions-root', help='Folder written by "track"; default: the stored predictions')
    s.add_argument('--output', default='results/reproduced')
    s.add_argument('--check', action='store_true', help='Compare with results/two_clip_experiment/completed')
    s.set_defaults(func=score)
    t = sub.add_parser('track')
    t.add_argument('--selected-weights')
    t.add_argument('--original-weights')
    t.add_argument('--output', default='runs/reproduced')
    t.set_defaults(func=track)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
