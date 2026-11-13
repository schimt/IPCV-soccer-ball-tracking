"""Regression checks for sparse manual review and full-frame paired inference."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from balltrack import annotate, evaluate, test_clip
from balltrack.common import digest, read_rows, reviewed_annotation, row, save_json, video_info, write_rows


def video(path, count=15, fps=25):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (80, 64))
    assert writer.isOpened()
    for i in range(count):
        writer.write(np.full((64, 80, 3), i % 255, dtype=np.uint8))
    writer.release()


@pytest.mark.parametrize('key,status', [('r', 'visible'), ('o', 'occluded'),
                                       ('u', 'uncertain'), ('f', 'out_of_frame')])
def test_review_advances_selected_frame_and_replaces_proposal_provenance(key, status):
    rows = [row(i, 25, 'visible', [10, 10, 4, 4], source='interpolated_linear') for i in [0, 5, 10]]
    value, next_index = annotate.review_action(rows, 0, key, 25)
    assert rows[next_index]['frame'] == 5
    assert value['status'] == status and reviewed_annotation(value, final=True)
    assert value['x'] == (10 if key == 'r' else '')
    assert rows[0]['reviewed'] == 0  # caller can record original provenance before commit
    rows[0] = value
    assert annotate.first_pending(rows) == 1
    _, last = annotate.review_action(rows, 2, key, 25)
    assert last == 2


def test_r_cannot_skip_unlabeled():
    with pytest.raises(ValueError, match='Choose a box'):
        annotate.review_action([row(0, 25, 'unlabeled')], 0, 'r', 25)


@pytest.mark.parametrize('source,reviewed', [('interpolated_linear', True),
                                          ('assistant_visual_keyframe', True), ('manual', False)])
def test_unreviewed_or_raw_interpolated_proposals_are_never_gt(source, reviewed):
    g = row(0, 25, 'visible', [10, 10, 4, 4], source=source, reviewed=reviewed)
    with pytest.raises(ValueError, match='manually reviewed'):
        evaluate.score_rows([g], [], 80, 64, 25, stride=5)


def mock_gui(monkeypatch, keys, draw_on_first=False):
    callback, seen = {}, []
    for name in ['namedWindow', 'imshow', 'destroyAllWindows']:
        monkeypatch.setattr(cv2, name, lambda *a: None)
    monkeypatch.setattr(cv2, 'setMouseCallback', lambda window, fn: callback.update(mouse=fn))
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *a: 1)
    iterator = iter(keys)
    def wait(delay):
        if draw_on_first and not seen:
            callback['mouse'](cv2.EVENT_LBUTTONDOWN, 100, 208, 0, None)
            callback['mouse'](cv2.EVENT_LBUTTONUP, 150, 258, 0, None)
        key = next(iterator)
        seen.append(key)
        return ord(key)
    monkeypatch.setattr(cv2, 'waitKey', wait)
    return seen


def test_gui_save_advance_quit_and_resume(tmp_path, monkeypatch):
    clip, csv = tmp_path/'clip.mp4', tmp_path/'annotations.csv'
    video(clip)
    rows, _ = annotate.initialize(clip, csv, stride=5)
    rows[0] = row(0, 25, 'visible', [1, 1, 4, 4], source='interpolated_linear')
    write_rows(csv, rows)
    args = SimpleNamespace(video=str(clip), output=str(csv), stride=5, init_only=False)
    mock_gui(monkeypatch, 'roq', draw_on_first=True)
    annotate.main(args)
    rows = read_rows(csv, 'gt')
    assert [r['frame'] for r in rows] == [0, 5, 10]
    assert rows[0]['reviewed'] == 1 and rows[0]['x'] > 1  # latest mouse box, not stale proposal
    assert rows[1]['status'] == 'occluded' and rows[1]['reviewed'] == 1
    assert rows[2]['reviewed'] == 0
    mock_gui(monkeypatch, 'fq')
    annotate.main(args)  # must resume at frame 10
    rows = read_rows(csv, 'gt')
    assert rows[2]['status'] == 'out_of_frame' and rows[2]['reviewed'] == 1
    audit = [json.loads(line) for line in csv.with_suffix('.audit.jsonl').read_text().splitlines()]
    assert [a['frame'] for a in audit] == [0, 0, 5, 10]
    assert audit[0]['before']['source'] == 'interpolated_linear'


def test_quit_saves_just_drawn_box_without_approval(tmp_path, monkeypatch):
    clip, csv = tmp_path/'clip.mp4', tmp_path/'annotations.csv'
    video(clip)
    mock_gui(monkeypatch, 'q', draw_on_first=True)
    annotate.main(SimpleNamespace(video=str(clip), output=str(csv), stride=5, init_only=False))
    rows, _ = annotate.initialize(clip, csv, stride=5)
    assert rows[0]['status'] == 'visible' and not rows[0]['reviewed']
    assert annotate.first_pending(rows) == 0


def test_sparse_loss_is_sampled_and_off_schedule_errors_do_not_count():
    gt = [row(i, 25, 'visible', [10, 10, 4, 4], source='manual', reviewed=True) for i in [0, 5, 10]]
    predictions = [row(i, 25, 'observed', [60, 50, 4, 4]) for i in range(15)]
    predictions[0] = row(0, 25, 'observed', [10, 10, 4, 4])
    predictions[5] = row(5, 25, 'missing')
    metrics, details, losses = evaluate.score_rows(gt, predictions, 80, 64, 25, stride=5)
    assert (metrics['tp'], metrics['fp'], metrics['fn']) == (1, 1, 2)
    assert [d['frame'] for d in details] == [0, 5, 10]
    assert losses == [dict(start_frame=5, annotated_frames=2, duration_seconds=.4)]


@pytest.mark.parametrize('stride', [1, 5])
def test_final_pair_scores_registered_schedule_but_requires_1500_predictions(tmp_path, monkeypatch, stride):
    clip, csv, registry = tmp_path/'clip.mp4', tmp_path/'annotations.csv', tmp_path/'registry.json'
    video(clip, 1500)
    meta = video_info(clip)
    save_json(registry, dict(clip=meta, annotation_stride=stride))
    monkeypatch.setattr(test_clip, 'REGISTRY', registry)
    meta.update(annotation_stride=stride, source='manual', role='sole_final_test')
    save_json(csv.with_suffix('.meta.json'), meta)
    gt = [row(i, 25, 'visible', [10, 10, 4, 4], source='manual_review', reviewed=True) for i in range(0, 1500, stride)]
    write_rows(csv, gt)
    predictions = [row(i, 25, 'observed', [10, 10, 4, 4] if i % stride == 0 else [60, 50, 4, 4]) for i in range(1500)]
    specs = []
    for method in ['classical', 'nn']:
        path = tmp_path/method/'predictions.csv'
        write_rows(path, predictions)
        save_json(path.with_name('run.json'), dict(method=method, video=meta,
                  timing=dict(pipeline_fps=10, end_to_end_fps=8)))
        specs.append(f'{method}={path}')
    args = SimpleNamespace(config='configs/default.yaml', gt=str(csv), predictions=specs,
                           output=str(tmp_path/'results'), require_test_minute=True)
    evaluate.main(args)
    result = json.loads((tmp_path/'results'/'metrics.json').read_text())
    assert result['annotation_count'] == 1500//stride and result['video_frames'] == 1500
    assert result['annotation_frames'] == list(range(0, 1500, stride))
    assert result['annotation_hz'] == 25/stride
    assert result['loss_duration_is_sampled_approximation'] == (stride > 1)
    assert len(result['metrics']) == 4
    assert all(m['tp'] == 1500//stride and m['fp'] == 0 for m in result['metrics'])
    write_rows(tmp_path/'nn'/'predictions.csv', predictions[::5])
    with pytest.raises(ValueError, match='every video frame'):
        evaluate.main(args)
    write_rows(tmp_path/'nn'/'predictions.csv', predictions)
    gt[0]['reviewed'] = 0
    write_rows(csv, gt)
    with pytest.raises(ValueError, match='manually reviewed'):
        evaluate.main(args)
    # Registered final-test metadata enforces the protocol even without the CLI flag.
    args.require_test_minute = False
    wrong_stride = 5 if stride == 1 else 1
    meta['annotation_stride'] = wrong_stride
    save_json(csv.with_suffix('.meta.json'), meta)
    write_rows(csv, [row(i, 25, 'out_of_frame', source='manual', reviewed=True) for i in range(0, 1500, wrong_stride)])
    with pytest.raises(ValueError, match='annotation schedule'):
        evaluate.main(args)
