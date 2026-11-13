from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from balltrack.common import config, row
from balltrack.development import load_cache, replay, validate_sequences
from balltrack.tracking import ClassicalDetector, BallTracker, Candidate


def test_validation_loader_rejects_nonvalidation_role_and_game(tmp_path):
    p = tmp_path / 'cache.json'
    for role, game in [('sole_final_test','9'), ('soccernet_validation','4')]:
        p.write_text(json.dumps(dict(role=role,sequences=[dict(metadata=dict(game=game))])))
        with pytest.raises(ValueError):
            load_cache(p)


def test_sequence_guard_rejects_training_game(tmp_path):
    p = tmp_path / 'SNMOT-060'
    p.mkdir()
    (p/'seqinfo.ini').write_text('[Sequence]\nname=SNMOT-060\nimWidth=100\nimHeight=100\nframeRate=25\nseqLength=2\nimDir=img1\n')
    (p/'gameinfo.ini').write_text('[Sequence]\ngameID=4\ntrackletID_1=ball;1\n')
    with pytest.raises(ValueError,match='reserves game 9'):
        validate_sequences(tmp_path,['SNMOT-060'],2)


def test_replay_resets_between_sequences_and_excludes_ambiguous_frames():
    cfg = config()
    meta = dict(name='first',game='9',width=100,height=100,fps=25)
    def frame(i, status, detections):
        return dict(gt=row(i,25,status,(10,10,6,6) if status=='visible' else None,
                           reviewed=True,source='SoccerNet_MOT_import'),
                    candidates=detections, affine=np.eye(2,3).tolist(),cut=False)
    hit = dict(box=[10,10,6,6],score=.9)
    data = dict(method='nn',sequences=[dict(metadata=meta,frames=[frame(0,'visible',[hit]),frame(1,'uncertain',[hit])]),
               dict(metadata=dict(meta,name='second'),frames=[frame(0,'visible',[])])])
    result = replay(data,cfg,.15)
    assert result['aggregate']['tp'] == 1
    assert result['aggregate']['fn'] == 1
    assert result['aggregate']['fp'] == 0
    assert result['per_sequence'][1]['visible_frames_with_position'] == 0


def test_field_prior_keeps_nongreen_ball_near_grass_and_rejects_other_background():
    cfg = deepcopy(config()['classical'])
    cfg.update(polarity='bright',min_field_context=.3,field_radius=25,min_area=3,
               brightness_weight=.35)
    image = np.full((160,240,3),60,np.uint8)
    image[:,:120] = (30,120,30)
    cv2.circle(image,(60,80),5,(230,230,230),-1)
    cv2.circle(image,(180,80),5,(230,230,230),-1)
    candidates = ClassicalDetector(cfg).detect(image)
    assert any(np.linalg.norm(c.center-[60,80]) < 5 for c in candidates)
    assert not any(np.linalg.norm(c.center-[180,80]) < 5 for c in candidates)


def test_optional_reacquisition_requires_margin_and_resets_velocity():
    cfg = deepcopy(config()['tracker'])
    cfg.update(reacquire_score_margin=.15,gate_diagonal_fraction=.02)
    tracker = BallTracker(cfg,1000,1000,25)
    local = Candidate((10,10,10,10),.5)
    tracker.step([local])
    distant = Candidate((800,800,10,10),.6)
    assert tracker.step([local,distant])[1] == local.box
    distant.score = .9
    assert tracker.step([local,distant])[1] == distant.box
    assert np.all(tracker.state[2:] == 0)


def test_blob_detector_retains_ball_and_suppresses_straight_line():
    cfg = deepcopy(config()['classical'])
    cfg.update(algorithm='hessian',blob_scales=[2.,3.,5.,7.],blob_threshold=5.,min_field_context=.25)
    image = np.zeros((160,240,3),np.uint8)
    image[:] = (30,100,30)
    cv2.circle(image,(60,80),7,(230,230,230),-1)
    cv2.line(image,(180,0),(180,159),(230,230,230),5)
    candidates = ClassicalDetector(cfg).detect(image)
    assert any(np.linalg.norm(d.center-[60,80]) < 3 for d in candidates)
    assert not any(abs(d.center[0]-180) < 5 for d in candidates)
