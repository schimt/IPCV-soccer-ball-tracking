import json
from types import SimpleNamespace

import pytest

from balltrack import two_clip
from balltrack.common import row, save_json, write_rows
from balltrack.evaluate import score_rows


def test_resolution_normalization_and_visibility_policy():
    def sample(scale):
        gt = [row(0,25,'visible',[100*scale,100*scale,4*scale,4*scale],source='manual',reviewed=True),
              row(1,25,'visible',[100*scale,100*scale,4*scale,4*scale],source='manual',reviewed=True),
              row(2,25,'occluded',source='manual',reviewed=True),
              row(3,25,'out_of_frame',source='manual',reviewed=True),
              row(4,25,'uncertain',source='manual',reviewed=True)]
        pred = [row(0,25,'observed',[110*scale,100*scale,4*scale,4*scale]),
                row(1,25,'missing'),row(2,25,'observed',[0,0,2,2]),
                row(3,25,'predicted',[0,0,2,2]),row(4,25,'observed',[0,0,2,2])]
        return score_rows(gt,pred,640*scale,360*scale,25)[0]
    small,large=sample(1),sample(2)
    assert (small['tp'],small['fp'],small['fn'])==(0,2,2)
    assert small['mean_center_error_px']*2 == large['mean_center_error_px']
    assert small['mean_center_error_normalized']==large['mean_center_error_normalized']
    assert small['visible_frames_with_position']==1
    assert small['invisible_predicted_positions_not_scored']==1
    assert small['uncertain_frames_excluded']==1


def test_reference_requires_manual_review_and_matching_video(tmp_path):
    path=tmp_path/'annotations.csv'
    info=dict(sha256='clip2',frames=2,fps=25,width=640,height=360)
    save_json(path.with_suffix('.meta.json'),dict(info,annotation_stride=1))
    values=[row(i,25,'visible',[10,10,4,4],source='assistant_visual_keyframe',reviewed=True) for i in range(2)]
    write_rows(path,values)
    with pytest.raises(ValueError,match='manual review'):
        two_clip.validate_reference(path,info)
    for r in values:
        r['source']='manual_review'
    write_rows(path,values)
    assert len(two_clip.validate_reference(path,info))==2
    with pytest.raises(ValueError,match='another clip'):
        two_clip.validate_reference(path,dict(info,sha256='clip1'))
    values[1]['cx']=19
    write_rows(path,values)
    with pytest.raises(ValueError,match='center/box'):
        two_clip.validate_reference(path,info)


def test_quality_requires_fresh_explicit_actions_and_current_values():
    rows=[row(i,25,'out_of_frame',source='manual',reviewed=True) for i in range(3)]
    # The real reader converts empty geometry to None.
    rows=[{k:None if v=='' else v for k,v in r.items()} for r in rows]
    plan=dict(created_utc='2026-10-19T12:00:00+00:00',frames=[0,2])
    events=[dict(time_utc='2026-10-19T12:01:00+00:00',action='set_status',frame=i,after=dict(rows[i])) for i in [0,2]]
    two_clip.validate_quality(rows,plan,events)
    with pytest.raises(ValueError,match='incomplete'):
        two_clip.validate_quality(rows,plan,events[:1])
    events[0]['time_utc']='2026-10-18T12:00:00+00:00'
    with pytest.raises(ValueError,match='incomplete'):
        two_clip.validate_quality(rows,plan,events)
    events[0]['time_utc']='2026-10-19T12:01:00+00:00'
    rows[0]['source']='manual_review'
    with pytest.raises(ValueError,match='changed after'):
        two_clip.validate_quality(rows,plan,events)


def test_evaluation_gate_precedes_inference_and_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(two_clip,'ROOT',tmp_path/'result')
    def blocked():
        raise ValueError('manual review incomplete')
    monkeypatch.setattr(two_clip,'check',blocked)
    monkeypatch.setattr(two_clip.subprocess,'run',lambda *a,**k:pytest.fail('Inference started before review'))
    with pytest.raises(ValueError,match='manual review'):
        two_clip.evaluate()
    assert not (tmp_path/'result').exists()


def test_quality_sample_contains_boundaries_and_is_repeatable():
    rows=[row(i,25,'visible' if i<20 else 'uncertain', [10,10,2,2] if i<20 else None,
              source='manual',reviewed=True) for i in range(30)]
    sample=two_clip.quality_frames(rows)
    assert {0,19,20,29} <= set(sample)
    assert sample==two_clip.quality_frames(rows)
