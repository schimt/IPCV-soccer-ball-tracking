"""Time the selected pipelines on the same SoccerNet JPEGs after training finishes."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .common import row, save_json, digest, write_rows
from .development import load_cache
from .evaluate import score_rows
from .tracking import ClassicalDetector, CameraMotion, BallTracker


def main():
    cv2.setNumThreads(2)
    record=json.loads(Path('runs/nn_expanded_v2/training_record.json').read_text())
    if record['status'] != 'completed':
        raise ValueError('Finish GPU training before the timing comparison')
    output=Path('results/development_v2/benchmark')
    if output.exists():
        raise ValueError('Preserve timing records: use a fresh output directory')
    output.mkdir()
    results=[]
    for method in ['classical','nn']:
        selection_path=Path('results/development_v2')/(method+'_final_selection')/'selection.json'
        selection=json.loads(selection_path.read_text()); cfg=selection['config']
        data=load_cache(selection['selected_cache'])
        if method=='nn':
            from .nn import NeuralDetector
            detector=NeuralDetector(cfg['nn'],cfg['nn']['weights'])
        else:
            detector=ClassicalDetector(cfg['classical'])
        detector.detect(cv2.imread(data['sequences'][0]['frames'][0]['image']))
        durations=[]; metrics=[]; begin_total=time.perf_counter()
        for sequence in data['sequences']:
            meta=sequence['metadata']; predictions=[]
            tracker=BallTracker(cfg['tracker'],meta['width'],meta['height'],meta['fps'])
            motion=CameraMotion(cfg['motion'])
            for item in sequence['frames']:
                frame=cv2.imread(item['image'])
                start=time.perf_counter()
                affine,cut=motion.update(frame)
                candidates=detector.detect(frame)
                status,box,score=tracker.step(candidates,affine,cut)
                durations.append(time.perf_counter()-start)
                predictions.append(row(item['gt']['frame'],meta['fps'],status,box,score,method))
            m,_,_=score_rows([f['gt'] for f in sequence['frames']],predictions,meta['width'],meta['height'],meta['fps'])
            metrics.append(dict(sequence=meta['name'],**m))
            write_rows(output/method/meta['name']/'predictions.csv',predictions)
        elapsed=time.perf_counter()-begin_total
        aggregate={k:sum(m[k] for m in metrics) for k in ['tp','fp','fn']}
        expected={k:selection['results']['aggregate'][k] for k in aggregate}
        result=dict(method=method,frames=len(durations),pipeline_fps=len(durations)/sum(durations),
                    jpeg_loop_fps=len(durations)/elapsed,median_frame_ms=float(np.median(durations)*1000),
                    per_sequence=metrics,counts=aggregate,selected_cached_counts=expected,
                    cached_counts_reproduced=aggregate==expected,selection_sha256=digest(selection_path),
                    measured='JPEG decode excluded from pipeline timing; camera motion, detection, transfer and tracking included',
                    excluded='model load, one warm-up, rendering, video encoding',opencv_threads=2)
        results.append(result)
        save_json(output/'timing.json',results)
        print(method,result['pipeline_fps'],'fps; counts',aggregate,'match',aggregate==expected,flush=True)
        del detector
    print('Saved live validation timing and replay consistency check',flush=True)


if __name__=='__main__':
    main()
