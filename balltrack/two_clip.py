"""Add the instructor holdout without changing any historical experiment artifacts.

Run from the project root: python -m balltrack.two_clip --help.
The inference packages are copied from the two archived code versions, not edited.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace

import cv2

from . import annotate
from .common import config, digest, read_rows, reviewed_annotation, save_json, video_info
from .metric_audit import write_csv

ROOT = Path('results/two_clip_experiment')
FROZEN = Path('runs/two_clip_frozen')
DATA = Path('data/test_2')
VIDEO = DATA / 'denmark_wales_clip_1min.mp4'
GT = DATA / 'annotations.csv'
REGISTRY = Path('configs/two_clip_experiment.json')
BASELINE = Path('baselines/20261006T070810Z_before_soccernet_expansion')
SELECTED = Path('runs/development_v2_frozen')
PROTOCOL = Path('docs/two_clip_protocol.md')
VERSIONS = ('selected', 'original')
METHODS = ('classical', 'nn')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def copy_checked(source, target, expected=None):
    source, target = Path(source), Path(target)
    if expected:
        require(digest(source) == expected, f'Archive checksum mismatch: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(digest(source) == digest(target), f'Refusing to overwrite {target}')
    else:
        shutil.copy2(source, target)
    require(digest(source) == digest(target), f'Copy mismatch: {target}')


def decoded_info(path):
    info = video_info(path)
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(frame.shape[:2] == (info['height'], info['width']), 'Changing frame dimensions')
            count += 1
    finally:
        cap.release()
    require(count == info['frames'], 'Incomplete video decoding')
    return dict(info, decoded_frames=count)


def verify():
    record = read(REGISTRY)
    for path, expected in record['protected_sha256'].items():
        require(Path(path).is_file() and digest(path) == expected, f'Protected artifact changed: {path}')
    for path, expected in record['frozen_sha256'].items():
        require(Path(path).is_file() and digest(path) == expected, f'Frozen experiment changed: {path}')
    for clip in record['clips']:
        require(digest(clip['video']['video']) == clip['video']['sha256'], 'Registered video changed')
    return record


def prepare():
    if REGISTRY.exists():
        verify()
        print('Already prepared; frozen files verified. Annotation work preserved.')
        return
    source = Path('new_video/denmark_wales_clip_1min.mp4')
    require(source.is_file(), f'Place the instructor video at {source.resolve()}')
    require(not FROZEN.exists(), 'Partial preparation exists; inspect before continuing')
    info = decoded_info(source)
    require((info['frames'], info['fps'], info['width'], info['height']) == (1500, 25, 640, 360),
            'Instructor clip differs from the declared 1500 frames / 25 fps / 640x360')
    selected, original = read(SELECTED/'manifest.json'), read(BASELINE/'snapshot.json')
    for version, archive, manifest in [('selected', SELECTED, selected), ('original', BASELINE, original)]:
        for name in ['common.py', 'tracking.py', 'pipeline.py', 'nn.py', 'evaluate.py']:
            key = 'balltrack\\' + name
            copy_checked(archive/'balltrack'/name, FROZEN/version/'balltrack'/name, manifest['files'][key])
        copy_checked(BASELINE/'balltrack/__init__.py', FROZEN/version/'balltrack/__init__.py',
                     original['files']['balltrack\\__init__.py'])
    for method in METHODS:
        copy_checked(SELECTED/f'{method}.yaml', FROZEN/'selected'/f'{method}.yaml',
                     selected['files'][f'{method}.yaml'])
        copy_checked(BASELINE/'configs/default.yaml', FROZEN/'original'/f'{method}.yaml',
                     original['files']['configs\\default.yaml'])
        require(config(FROZEN/'original'/f'{method}.yaml') == read(Path('runs/final')/method/'run.json')['config'],
                'Original configuration does not match historical inference')
        require(config(FROZEN/'selected'/f'{method}.yaml') == read(Path('runs/final_selected_v2')/method/'run.json')['config'],
                'Selected configuration does not match historical inference')
    copy_checked(SELECTED/'nn_best.pt', FROZEN/'selected/nn_best.pt', selected['files']['nn_best.pt'])
    old_run = read('runs/final/nn/run.json')
    copy_checked(BASELINE/'runs/nn_pilot/fit/weights/best.pt', FROZEN/'original/nn_best.pt',
                 old_run['nn']['weights_sha256'])
    copy_checked(source, VIDEO)
    rows, meta = annotate.initialize(VIDEO, GT, stride=1)
    meta.update(role='test_clip_2', label='Test clip 2: instructor Denmark-Wales',
                source='Instructor-provided video; independent manual annotation',
                forbidden_uses=['training', 'validation', 'parameter_selection'])
    save_json(GT.with_suffix('.meta.json'), meta)
    save_json(GT.with_suffix('.progress.json'), annotate.annotation_progress(rows))
    for name in ['report.md', 'report.pdf']:
        copy_checked(Path('report')/name, ROOT/'before_extension'/name)
    copy_checked('README.md', ROOT/'before_extension/README.md')
    copy_checked('docs/protocol.md', ROOT/'before_extension/protocol.md')
    protected = set()
    for folder in ['data/test', 'runs/final', 'runs/final_selected_v2', 'runs/development_v2_frozen',
                   'results/test', 'results/test_reviewed', 'results/final_selected_v2', 'baselines']:
        protected.update(p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protected.update(Path('runs').rglob('*.pt'))
    protected.update(Path('configs').glob('*.yaml'))
    protected.add(Path('configs/final_test.json'))
    clips = [dict(id='test_1', label='Test clip 1: own clip', previously_evaluated=True,
                  video=video_info('data/test/clip.mp4'), annotations='results/final_selected_v2/reference/annotations.csv'),
             dict(id='test_2', label='Test clip 2: instructor Denmark-Wales', previously_evaluated=False,
                  video=dict(video_info(VIDEO), decoded_frames=info['decoded_frames']), annotations=str(GT))]
    record = dict(frozen_utc=utc(), status='awaiting_manual_review', clips=clips,
                  source_video=str(source), source_sha256=digest(source),
                  primary_comparison='Validation-selected classical and YOLOv8n pipelines on each clip separately',
                  versions=list(VERSIONS), methods=list(METHODS),
                  selection_source='Existing SoccerNet validation selections; no test-driven tuning',
                  protocol=str(PROTOCOL),
                  protected_sha256={str(p):digest(p) for p in sorted(protected)},
                  frozen_sha256={str(p):digest(p) for p in sorted(FROZEN.rglob('*')) if p.is_file()})
    for path in [PROTOCOL, Path(__file__), Path('balltrack/annotate.py'), Path('balltrack/metric_audit.py')]:
        record['frozen_sha256'][str(path)] = digest(path)
    save_json(REGISTRY, record)
    save_json(ROOT/'freeze.json', record)
    print('Prepared test clip 2: 1500 unlabeled, unreviewed frames. No model inference or scoring performed.')


def validate_reference(path, info):
    path = Path(path)
    meta, rows = read(path.with_suffix('.meta.json')), read_rows(path, 'gt')
    require(meta['sha256'] == info['sha256'], 'Reference belongs to another clip')
    require(all(meta[k] == info[k] for k in ['frames', 'fps', 'width', 'height']), 'Reference metadata mismatch')
    require(meta['annotation_stride'] == 1 and [g['frame'] for g in rows] == list(range(info['frames'])),
            'Reference must contain every frame at stride 1')
    require(all(reviewed_annotation(g, final=True) for g in rows),
            'Evaluation blocked: complete manual review of all 1500 frames first')
    for g in rows:
        require(abs(g['timestamp']-g['frame']/info['fps']) < 1e-6, 'Reference timestamp mismatch')
        if g['status'] == 'visible':
            x,y,w,h = (g[k] for k in ['x','y','w','h'])
            require(all(v is not None for v in [x,y,w,h]) and w > 0 and h > 0
                    and 0 <= x < x+w <= info['width'] and 0 <= y < y+h <= info['height'],
                    f'Invalid reference box at frame {g["frame"]}')
            require(abs(g['cx']-(x+w/2)) < 1e-6 and abs(g['cy']-(y+h/2)) < 1e-6,
                    'Reference center/box mismatch')
    return rows


def quality_boundaries(rows):
    """Visibility changes can require extra review after annotation corrections."""
    frames = {0, len(rows)-1}
    for a,b in zip(rows,rows[1:]):
        if a['status'] != b['status']:
            frames.update([a['frame'], b['frame']])
    return frames


def quality_frames(rows):
    """Initial sample only: boundaries + seeded 10% visible, without predictions."""
    frames = quality_boundaries(rows)
    visible = [g['frame'] for g in rows if g['status']=='visible']
    frames.update(random.Random(42).sample(visible, math.ceil(len(visible)*.1)))
    return sorted(frames)


def required_quality_frames(rows, plan):
    """Keep the recorded sample fixed; never redraw it after changing a label."""
    return set(plan['frames']) | quality_boundaries(rows)


def quality_plan():
    record = verify()
    rows = validate_reference(GT, record['clips'][1]['video'])
    target = DATA/'quality_plan.json'
    if target.exists():
        print('Existing quality plan preserved:', target)
        return
    frames = quality_frames(rows)
    save_json(target, dict(created_utc=utc(), annotation_sha256=digest(GT), frames=frames,
                           rule='All status boundaries, first/last frame, random 10% visible (seed 42)'))
    text = '# Test clip 2: manual quality recheck\n\nInspect raw video; confirm with R or correct with box/O/U/F.\n\n| Frame | Reason |\n| --- | --- |\n'
    text += '\n'.join(f'| {i} | Predeclared quality sample |' for i in frames)+'\n'
    (DATA/'quality_review.md').write_text(text, encoding='utf-8')
    print(f'Quality queue ready: {len(frames)} frames. Use Annotate-Test2.ps1 -QualityReview.')


def validate_quality(rows, plan, events):
    reviewed = {}
    for e in events:
        if e.get('time_utc','') >= plan['created_utc'] and e.get('action') in {'human_review','set_status'}:
            reviewed[e['frame']] = e['after']
    required = required_quality_frames(rows, plan)
    require(required <= reviewed.keys(),
            'Quality review incomplete; confirm every listed frame (R/O/U/F), including any new visibility boundaries')
    for i in required:
        after = reviewed[i]
        for k in ['status','source','reviewed','cx','cy','x','y','w','h']:
            require((after.get(k) if after.get(k) != '' else None) == rows[i][k],
                    f'Frame {i} changed after its quality confirmation')


def check(quality=True):
    record = verify()
    for clip in record['clips']:
        rows = validate_reference(clip['annotations'], clip['video'])
        print(clip['label'], dict(Counter(g['status'] for g in rows)))
    if quality:
        require((DATA/'quality_plan.json').exists(), 'Run quality-plan and the manual recheck before evaluation')
        plan = read(DATA/'quality_plan.json')
        audit = GT.with_suffix('.audit.jsonl')
        events = [json.loads(line) for line in audit.read_text(encoding='utf-8').splitlines()] if audit.exists() else []
        # If corrections introduce new transitions, expand the readable queue without discarding earlier confirmations.
        extra = required_quality_frames(rows, plan) - set(plan['frames'])
        if extra:
            with (DATA/'quality_review.md').open('a',encoding='utf-8') as f:
                f.write('\n'.join(f'| {i} | New boundary/sample after correction |' for i in sorted(extra))+'\n')
            plan['frames'] = sorted(set(plan['frames']) | extra)
            save_json(DATA/'quality_plan.json', plan)
        validate_quality(rows, plan, events)
    return record


def frozen_module(name):
    alias = '_two_clip_scoring'
    if alias not in sys.modules:
        package = FROZEN/'selected/balltrack'
        spec = importlib.util.spec_from_file_location(alias, package/'__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    return importlib.import_module(alias+'.'+name)


def validate_predictions(path, info):
    run, rows = read(Path(path).with_name('run.json')), read_rows(path,'pred')
    require(run['video']['sha256'] == info['sha256'] and run['frames']==info['frames'], 'Wrong/incomplete prediction video')
    require([p['frame'] for p in rows] == list(range(info['frames'])), 'Predictions must include every frame')
    require(all(abs(p['timestamp']-p['frame']/info['fps']) < 1e-6 for p in rows), 'Prediction timestamp mismatch')
    return run, rows


def gt_video(clip, predictions, output):
    """Separate error-analysis render; never part of inference or measured FPS."""
    info = clip['video']
    gt = read_rows(clip['annotations'],'gt')
    cap = cv2.VideoCapture(info['video'])
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), info['fps'], (info['width'],info['height']))
    require(writer.isOpened(), 'Cannot create ground-truth visualization')
    draw = frozen_module('pipeline').draw_prediction
    try:
        for g,p in zip(gt,predictions):
            ok, frame = cap.read()
            require(ok, 'Incomplete ground-truth visualization')
            draw(frame,p,'Prediction')
            if g['status']=='visible':
                x,y,w,h = (g[k] for k in ['x','y','w','h'])
                cv2.rectangle(frame,(round(x),round(y)),(round(x+w),round(y+h)),(255,255,0),1)
                cv2.circle(frame,(round(g['cx']),round(g['cy'])),4,(255,255,0),1)
            cv2.putText(frame,'GT: '+g['status']+' | cyan=reference',(8,55),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,0),1)
            writer.write(frame)
    finally:
        cap.release(); writer.release()


def evaluate():
    record = check()  # Must pass BEFORE any inference, output directory, or metric computation.
    out = ROOT/'completed'
    runroot = Path('runs/two_clip_experiment/test_2')
    require(not out.exists() and not runroot.exists(), 'Evaluation already started; inspect preserved outputs before retrying')
    out.mkdir(parents=True)
    copy_checked(GT,out/'reference/annotations.csv')
    copy_checked(GT.with_suffix('.meta.json'),out/'reference/annotations.meta.json')
    for file in ['quality_plan.json','quality_review.md','annotations.audit.jsonl']:
        copy_checked(DATA/file,out/'reference'/file)
    reference_hash = digest(GT)
    save_json(out/'start.json',dict(started_utc=utc(),registry_sha256=digest(REGISTRY),annotation_sha256=reference_hash))
    # Isolated archived packages keep original and selected tracker implementations distinct.
    for version in VERSIONS:
        for method in METHODS:
            args = dict(video=str(VIDEO),method=method,config=str(FROZEN/version/f'{method}.yaml'),
                        output=str(runroot/version/method),weights=str(FROZEN/version/'nn_best.pt') if method=='nn' else None,
                        pretrained=False)
            code = ('import sys,json,cv2; from types import SimpleNamespace; '
                    'sys.path.insert(0,sys.argv[1]); from balltrack.pipeline import run; '
                    'cv2.setNumThreads(2); run(SimpleNamespace(**json.loads(sys.argv[2])))')
            subprocess.run([sys.executable,'-B','-c',code,str((FROZEN/version).resolve()),json.dumps(args)],check=True)
    require(digest(GT)==reference_hash, 'Annotations changed during inference; evaluation stopped')
    metrics = []
    evaluator = frozen_module('evaluate')
    for clip in record['clips']:
        folder = out/clip['id']; folder.mkdir()
        info = clip['video']; gt = validate_reference(clip['annotations'],info)
        for version in VERSIONS:
            for method in METHODS:
                source = ((Path('runs/final_selected_v2') if version=='selected' else Path('runs/final'))/method
                          if clip['id']=='test_1' else runroot/version/method)
                run,pred = validate_predictions(source/'predictions.csv',info)
                require(run['method']==method and run['config']==config(FROZEN/version/f'{method}.yaml'), 'Prediction configuration mismatch')
                if method=='nn':
                    require(run['nn']['weights_sha256']==digest(FROZEN/version/'nn_best.pt'), 'Prediction weights mismatch')
                name = method+'_'+version
                for observed_only in [False, True]:
                    m,details,losses = evaluator.score_rows(gt,pred,info['width'],info['height'],info['fps'],.01,1,observed_only)
                    mode = 'observed_only' if observed_only else 'tracking'
                    m.update(method=method,version=version,test_clip=clip['id'],mode=mode,status='evaluated',
                             visible_frames_without_position=m['visible_frames']-m['visible_frames_with_position'],
                             missing_predictions_all_frames=sum(p['status']=='missing' for p in pred),
                             pipeline_fps=run['timing']['pipeline_fps'],end_to_end_fps=run['timing']['end_to_end_fps'],
                             timing_source='historical' if clip['id']=='test_1' else 'new sequential run; OpenCV threads=2',
                             predictions=str(source/'predictions.csv'),predictions_sha256=digest(source/'predictions.csv'))
                    metrics.append(m)
                    write_csv(folder/f'{name}_{mode}_frames.csv',details)
                    save_json(folder/f'{name}_{mode}_lost_periods.json',losses)
                copy_checked(source/'run.json',folder/f'{name}_run.json')
                gt_video(clip,pred,folder/f'{name}_with_ground_truth.mp4')
        save_json(folder/'reference_membership.json',dict(status_counts=dict(Counter(g['status'] for g in gt)),
                  excluded_uncertain=[g['frame'] for g in gt if g['status']=='uncertain'],annotation_sha256=digest(clip['annotations'])))
    verify()
    require(digest(GT)==reference_hash, 'Annotations changed during evaluation')
    save_json(out/'metrics.json',dict(dataset_source='Two separate test clips; frozen methods',metrics=metrics))
    write_csv(out/'comparison_all_modes.csv',metrics)
    write_csv(out/'comparison.csv',[m for m in metrics if m['mode']=='tracking'])
    save_json(out/'completion.json',dict(completed_utc=utc(),annotation_sha256=reference_hash,
              registry_sha256=digest(REGISTRY),status='quantitative evaluation complete; visual error analysis pending',
              outputs_sha256={str(p):digest(p) for folder in [out,runroot] for p in folder.rglob('*') if p.is_file()}))
    print('Saved separate metrics and GT videos:',out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','verify','check','quality-plan','evaluate'])
    args = parser.parse_args()
    {'prepare':prepare,'verify':verify,'check':check,'quality-plan':quality_plan,'evaluate':evaluate}[args.action]()


if __name__=='__main__':
    main()
