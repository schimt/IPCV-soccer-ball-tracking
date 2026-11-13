import pytest

from balltrack.common import row
from balltrack.metric_audit import ledger


@pytest.mark.parametrize('observed_only,expected', [(False,(2,2,2,3)),(True,(1,2,3,2))])
def test_metric_membership_visibility_and_prediction_policy(observed_only,expected):
    statuses=['visible']*4+['occluded','out_of_frame','uncertain']
    gt=[row(i,25,s,(10,10,4,4) if s=='visible' else None,source='manual',reviewed=True)
        for i,s in enumerate(statuses)]
    pred=[row(i,25,s,box) for i,(s,box) in enumerate([
        ('observed',(10,10,4,4)),('predicted',(10,10,4,4)),
        ('observed',(80,80,4,4)),('missing',None),
        ('predicted',(10,10,4,4)),('observed',(10,10,4,4)),('observed',(10,10,4,4))])]
    rows,metrics=ledger(gt,pred,100,100,25,observed_only)
    assert tuple(metrics[k] for k in ['tp','fp','fn','visible_frames_with_position'])==expected
    assert rows[4]['invisible_prediction_not_scored']==1 and rows[4]['fp']==0
    assert rows[5]['fp']==1 and rows[5]['error_included']==0
    assert rows[6]['excluded_uncertain']==1
    assert sum(rows[6][k] for k in ['tp','fp','fn','error_included','penalty_included'])==0
    assert sum(r['penalty_included'] for r in rows)==4
    assert sum(r['pipeline_processed'] for r in rows)==7


def test_metric_ledger_rejects_missing_prediction_row():
    gt=[row(0,25,'uncertain',source='manual',reviewed=True)]
    with pytest.raises(AssertionError):
        ledger(gt,[],100,100,25)
