"""Explicit per-frame metric membership; does not change the frozen evaluator."""
from collections import Counter
import csv
import json
import math
from pathlib import Path

from .common import digest, read_rows, save_json
from .data import ball_boxes, sequence_metadata
from .evaluate import score_rows


def ledger(gt, predictions, width, height, fps, observed_only=False, sequence='test'):
    assert [r['frame'] for r in predictions] == [g['frame'] for g in gt]
    diagonal=math.hypot(width,height); threshold=.01*diagonal
    rows=[]
    for g,p in zip(gt,predictions):
        visible=g['status']=='visible'; ignored=g['status']=='uncertain'
        accepted=p['status'] in ({'observed'} if observed_only else {'observed','predicted'})
        error=math.hypot(p['cx']-g['cx'],p['cy']-g['cy']) if visible and accepted else None
        tp=int(error is not None and error<=threshold)
        fn=int(visible and not tp)
        fp=int((visible and accepted and not tp) or
               (not visible and not ignored and p['status']=='observed'))
        penalty=(min(error/diagonal,1) if error is not None else 1) if visible else None
        rows.append(dict(sequence=sequence,frame=g['frame'],timestamp=g['timestamp'],
            gt_status=g['status'],prediction_status=p['status'],prediction_cx=p['cx'],prediction_cy=p['cy'],
            gt_cx=g['cx'],gt_cy=g['cy'],pipeline_processed=1,excluded_uncertain=int(ignored),
            tp=tp,fp=fp,fn=fn,precision_denominator=tp+fp,recall_denominator=int(visible),
            f1_denominator=2*tp+fp+fn,error_included=int(error is not None),center_error_px=error,
            penalty_included=int(visible),capped_normalized_error_with_missing_penalty=penalty,
            loss_frame=fn,loss_boundary=int(not visible),
            invisible_prediction_not_scored=int(not visible and not ignored and p['status']=='predicted')))
    metrics,_,_=score_rows(gt,predictions,width,height,fps,observed_only=observed_only)
    for key in ['tp','fp','fn']:
        assert sum(r[key] for r in rows)==metrics[key]
    assert sum(r['error_included'] for r in rows)==metrics['visible_frames_with_position']
    assert sum(r['recall_denominator'] for r in rows)==metrics['visible_frames']
    assert sum(r['excluded_uncertain'] for r in rows)==metrics['uncertain_frames_excluded']
    errors=[r['center_error_px'] for r in rows if r['error_included']]
    if errors:
        assert math.isclose(sum(errors)/len(errors),metrics['mean_center_error_px'])
    return rows,metrics


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,rows[0].keys());writer.writeheader();writer.writerows(rows)


def validation_audit():
    root=Path('results/development_v2/membership_audit');root.mkdir(exist_ok=True)
    cache=json.loads(Path('results/development_v2/baseline_classical_cache.json').read_text())
    membership=[];source_summary=[];provenance={}
    for seq in cache['sequences']:
        name=seq['metadata']['name'];folder=Path('data/soccernet_v2/train')/name
        meta=sequence_metadata(folder);boxes=ball_boxes(folder,meta)
        assert digest(folder/'gt/gt.txt')==seq['annotation_sha256']
        provenance[name]={str(folder/file):digest(folder/file) for file in ['gt/gt.txt','gameinfo.ini','seqinfo.ini']}
        raw={i:[] for i in range(300)}
        with (folder/'gt/gt.txt').open(newline='') as f:
            for values in csv.reader(f):
                frame,track=int(values[0])-1,int(values[1])
                if frame in raw and track in meta['ball_ids']:
                    raw[frame].append([float(v) for v in values])
        counts=Counter();raw_invalid=0
        for i,frame in enumerate(seq['frames']):
            assert frame['gt']['frame']==i
            valid=boxes.get(i,[]);count=len(valid)
            status='visible' if count==1 else 'uncertain'
            assert frame['gt']['status']==status
            if count==1:
                assert tuple(frame['gt'][k] for k in ['x','y','w','h'])==valid[0]
            reason='included_single_ball' if count==1 else 'excluded_no_valid_ball' if count==0 else 'excluded_multiple_balls'
            counts[reason]+=1;raw_invalid+=len(raw[i])-count
            membership.append(dict(sequence=name,game=meta['game'],frame=i,mot_frame=i+1,
                image=f'{meta["imdir"]}/{i+1:06d}.jpg',raw_ball_records=len(raw[i]),
                raw_ball_track_ids=';'.join(str(int(v[1])) for v in raw[i]),
                nonpositive_confidence_records=sum(v[6]<=0 for v in raw[i]),
                valid_ball_boxes=count,reason=reason,gt_status=status,
                included_in_localization_metrics=int(count==1),included_in_pipeline_timing=1,
                valid_boxes_json=json.dumps(valid)))
        source_summary.append(dict(sequence=name,processed=300,**counts,discarded_raw_ball_records=raw_invalid))
    write_csv(root/'validation_frame_membership.csv',membership)
    write_csv(root/'excluded_frames.csv',[r for r in membership if not r['included_in_localization_metrics']])
    variants={
        'Classical_baseline':'results/development_v2/classical_videos/baseline_predictions',
        'Classical_selected':'results/development_v2/classical_final_selection/predictions',
        'NN_pilot_baseline':'results/development_v2/nn_videos/baseline_predictions',
        'NN_pilot_retuned':'results/development_v2/pilot_nn_selection/predictions',
        'NN_selected':'results/development_v2/nn_final_selection/predictions'}
    recorded=json.loads(Path('results/development_v2/metrics.json').read_text())['metrics']
    summaries=[]
    for method,folder in variants.items():
        for mode in ['tracking','observed_only']:
            rows=[];per_seq=[]
            for seq in cache['sequences']:
                meta=seq['metadata'];path=Path(folder)/meta['name']/'predictions.csv'
                current,m=ledger([f['gt'] for f in seq['frames']],read_rows(path,'pred'),
                    meta['width'],meta['height'],meta['fps'],mode=='observed_only',meta['name'])
                rows.extend(current);per_seq.append(dict(sequence=meta['name'],**m,predictions=str(path),sha256=digest(path)))
            counts={k:sum(r[k] for r in rows) for k in ['tp','fp','fn','error_included','excluded_uncertain','invisible_prediction_not_scored']}
            if mode=='tracking':
                expected=next(m for m in recorded if m['method']==method)
                assert all(counts[k]==expected[k] for k in ['tp','fp','fn'])
            write_csv(root/f'{method}_{mode}.csv',rows)
            summaries.append(dict(method=method,mode=mode,**counts,per_sequence=per_seq))
    summary=dict(processed_frames=len(membership),included=sum(r['included_in_localization_metrics'] for r in membership),
        excluded=sum(not r['included_in_localization_metrics'] for r in membership),sequences=source_summary,
        annotation_provenance=provenance,metrics=summaries,
        rule='Exactly one valid source ball box; source presence does not establish manual visibility.',
        excluded_temporal_effect='Excluded frames still update camera motion and tracker; they may affect later outputs.',
        no_post_test_rule_change=True)
    save_json(root/'summary.json',summary)
    print(json.dumps(source_summary,indent=2),flush=True)


def final_audit():
    root=Path('results/final_selected_v2');gt=read_rows(root/'reference/annotations.csv','gt')
    variants={f'{method}_{version}':Path(folder)/method/'predictions.csv'
              for version,folder in [('original','runs/final'),('selected','runs/final_selected_v2')]
              for method in ['classical','nn']}
    summary=[]
    for name,path in variants.items():
        for mode in ['tracking','observed_only']:
            rows,metrics=ledger(gt,read_rows(path,'pred'),1350,1080,25,mode=='observed_only')
            write_csv(root/'membership'/f'{name}_{mode}.csv',rows)
            run=json.loads(path.with_name('run.json').read_text())
            metrics.update(method=name,mode=mode,predictions=str(path),predictions_sha256=digest(path),
                pipeline_fps=run['timing']['pipeline_fps'],end_to_end_fps=run['timing']['end_to_end_fps'])
            summary.append(metrics)
    write_csv(root/'comparison_all_modes.csv',summary)
    write_csv(root/'comparison.csv',[m for m in summary if m['mode']=='tracking'])
    save_json(root/'comparison.json',dict(dataset_source='Previously evaluated sole 60-second final test; fixed selection from SoccerNet',
        annotation_sha256=digest(root/'reference/annotations.csv'),metrics=summary))
    print(json.dumps([m for m in summary if m['mode']=='tracking'],indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['validation','final'])
    args=parser.parse_args()
    validation_audit() if args.dataset=='validation' else final_audit()
