import configparser
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from balltrack.common import config, read_rows, row, write_rows
from balltrack.data import ball_boxes, sequence_metadata
from balltrack.evaluate import score_rows
from balltrack.tracking import BallTracker, Candidate, ClassicalDetector
from balltrack.annotate import initialize
from balltrack.data import convert
from types import SimpleNamespace


def test_metadata_drives_ball_id_and_coordinate_conversion(tmp_path):
    (tmp_path / 'seqinfo.ini').write_text('[Sequence]\nname=example\nimWidth=100\nimHeight=80\nframeRate=25\nseqLength=3\nimDir=img1\n')
    (tmp_path / 'gameinfo.ini').write_text('[Sequence]\ngameID=42\ntrackletID_1=player team left;9\ntrackletID_37=ball;1\n')
    (tmp_path / 'gt').mkdir()
    (tmp_path / 'gt/gt.txt').write_text('1,1,5,5,10,10,1,-1,-1,-1\n2,37,11,21,8,6,1,-1,-1,-1\n')
    meta = sequence_metadata(tmp_path)
    assert meta['ball_ids'] == [37]
    assert ball_boxes(tmp_path, meta) == {1: [(10., 20., 8., 6.)]}


def test_missing_metadata_never_guesses_ball(tmp_path):
    with pytest.raises(ValueError, match='Missing metadata'):
        sequence_metadata(tmp_path)


def test_metrics_penalize_wrong_and_missing_localization_and_separate_extrapolation():
    gt = [row(0, 25, 'visible', (0, 0, 2, 2), reviewed=True),
          row(1, 25, 'visible', (0, 0, 2, 2), reviewed=True),
          row(2, 25, 'visible', (0, 0, 2, 2), reviewed=True),
          row(3, 25, 'out_of_frame', reviewed=True),
          row(4, 25, 'occluded', reviewed=True), row(5, 25, 'uncertain', reviewed=True)]
    pred = [row(0, 25, 'predicted', (0, 0, 2, 2)), row(1, 25, 'observed', (70, 0, 2, 2)),
            row(2, 25, 'missing'), row(3, 25, 'observed', (0, 0, 2, 2)),
            row(4, 25, 'predicted', (0, 0, 2, 2)), row(5, 25, 'observed', (0, 0, 2, 2))]
    m, _, losses = score_rows(gt, pred, 100, 100, 25)
    assert (m['tp'], m['fp'], m['fn']) == (1, 2, 2)
    assert m['recall'] == pytest.approx(1/3)
    assert m['mean_center_error_px'] == 35
    assert m['mean_capped_normalized_error_missing_penalty_1'] > .49
    assert m['invisible_predicted_positions_not_scored'] == 1
    assert losses == [dict(start_frame=1, annotated_frames=2, duration_seconds=.08)]
    observed, _, _ = score_rows(gt, pred, 100, 100, 25, observed_only=True)
    assert observed['tp'] == 0 and observed['fn'] == 3


def test_all_missing_is_zero_recall_not_good_error():
    g = [row(0, 25, 'visible', (0, 0, 2, 2), reviewed=True)]
    m, _, _ = score_rows(g, [], 100, 100, 25)
    assert m['recall'] == 0 and m['mean_center_error_px'] is None
    assert m['mean_capped_normalized_error_missing_penalty_1'] == 1


def test_unreviewed_ground_truth_rejected():
    with pytest.raises(ValueError, match='reviewed'):
        score_rows([row(0, 25, 'visible', (1, 1, 2, 2))], [], 100, 100, 25)


def test_tracker_expires_gap_and_resets_on_camera_cut():
    c = config()['tracker']
    t = BallTracker(c, 640, 360, 25)
    d = Candidate((100, 100, 10, 10), .9)
    assert t.step([d])[0] == 'observed'
    assert t.step([])[0] == 'predicted'
    for _ in range(t.max_gap):
        result = t.step([])
    assert result[0] == 'missing'
    t.step([d])
    assert t.step([], cut=True)[0] == 'missing'


@pytest.mark.parametrize('color', [20, 240])
def test_classical_detects_dark_and_bright_ball_without_nn(color):
    image = np.full((100, 160, 3), 110, dtype=np.uint8)
    cv2.circle(image, (80, 50), 4, (color, color, color), -1)
    boxes = ClassicalDetector(config()['classical']).detect(image)
    assert any(np.linalg.norm(d.center - np.array([80, 50])) < 4 for d in boxes)


def test_csv_refuses_duplicate_frames_and_invisible_geometry(tmp_path):
    p = tmp_path / 'pred.csv'
    write_rows(p, [row(0, 25, 'missing'), row(0, 25, 'missing')])
    with pytest.raises(ValueError, match='duplicate'):
        read_rows(p, 'pred')
    write_rows(p, [row(0, 25, 'occluded', (1, 1, 2, 2))])
    with pytest.raises(ValueError):
        read_rows(p, 'gt')


def test_nonfinite_timestamp_rejected(tmp_path):
    r = row(0, 25, 'missing')
    r['timestamp'] = 'nan'
    path = tmp_path / 'nan.csv'
    write_rows(path, [r])
    with pytest.raises(ValueError, match='timestamp'):
        read_rows(path, 'pred')


def test_annotation_resume_and_video_identity(tmp_path):
    def video(name, value):
        path = tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (64, 48))
        assert writer.isOpened()
        for _ in range(3):
            writer.write(np.full((48, 64, 3), value, np.uint8))
        writer.release()
        return path
    first, second = video('one.mp4', 20), video('two.mp4', 220)
    csv_path = tmp_path / 'gt.csv'
    rows, _ = initialize(first, csv_path)
    assert len(rows) == 3 and all(r['status'] == 'unlabeled' for r in rows)
    rows[0] = row(0, 25, 'visible', (10, 10, 4, 4), source='manual', reviewed=True)
    write_rows(csv_path, rows)
    resumed, _ = initialize(first, csv_path)
    assert resumed[0]['status'] == 'visible'
    with pytest.raises(ValueError, match='another video'):
        initialize(second, csv_path)


def test_converter_groups_entire_games_and_records_source(tmp_path):
    root = tmp_path / 'raw'
    for sequence, game in [('A', '4'), ('B', '4'), ('C', '9')]:
        folder = root / sequence
        (folder / 'gt').mkdir(parents=True)
        (folder / 'img1').mkdir()
        (folder / 'seqinfo.ini').write_text(f'[Sequence]\nname={sequence}\nimWidth=64\nimHeight=48\nframeRate=25\nseqLength=1\nimDir=img1\n')
        (folder / 'gameinfo.ini').write_text(f'[Sequence]\ngameID={game}\ntrackletID_7=ball;1\n')
        (folder / 'gt/gt.txt').write_text('1,7,11,11,5,5,1,-1,-1,-1\n')
        cv2.imwrite(str(folder / 'img1/000001.jpg'), np.zeros((48, 64, 3), np.uint8))
    args = SimpleNamespace(root=root, output=tmp_path/'yolo', stride=1, seed=42,
                           val_games=['9'], mot_origin=1, include_unlabeled_negatives=False)
    convert(args)
    import json
    manifest = json.loads((args.output / 'manifest.json').read_text())
    assert manifest['counts'] == {'train': 2, 'val': 1}
    assert {s['split'] for s in manifest['sequences'] if s['game'] == '4'} == {'train'}
    assert all(s['frame_indices'] == [0] and len(s['annotation_sha256']) == 64 for s in manifest['sequences'])
    with pytest.raises(ValueError, match='empty output'):
        convert(args)
