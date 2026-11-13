from balltrack.common import row
from balltrack.test2_review import pending_frames


def fixture():
    rows=[row(i,25,'out_of_frame',source='manual',reviewed=True) for i in range(4)]
    rows=[{k: None if v=='' else v for k,v in r.items()} for r in rows]
    plan=dict(created_utc='2026-10-19T13:00:00+00:00',frames=[0,1,3])
    event=dict(frame=0,action='set_status',time_utc='2026-10-19T13:01:00+00:00',after=dict(rows[0]))
    return rows,plan,event


def test_resume_skips_only_fresh_matching_confirmations():
    rows,plan,e=fixture()
    assert pending_frames(rows,plan,[e])==[1,3]
    e['time_utc']='2026-10-19T12:59:00+00:00'
    assert pending_frames(rows,plan,[e])==[0,1,3]


def test_priority_is_subset_and_edits_require_new_confirmation():
    rows,plan,e=fixture()
    assert pending_frames(rows,plan,[e],[0,1])==[1]
    rows[0]['status']='uncertain'
    assert pending_frames(rows,plan,[e],[0,1])==[0,1]


def test_new_status_boundaries_are_included_without_resetting_existing_approvals():
    rows,plan,e=fixture()
    rows[2]['status']='uncertain'
    assert pending_frames(rows,plan,[e])==[1,2,3]


def test_visibility_correction_keeps_initial_random_sample_fixed():
    from balltrack.two_clip import quality_frames, quality_boundaries, required_quality_frames
    rows=[row(i,25,'visible',[10,10,4,4],source='manual_review',reviewed=True) for i in range(200)]
    plan=dict(frames=quality_frames(rows))
    rows[30]=row(30,25,'occluded',source='manual',reviewed=True)
    required=required_quality_frames(rows,plan)
    assert required == set(plan['frames']) | {29,30,31}
    assert set(quality_frames(rows))-required  # A fresh random draw would add unrelated work.
    plan['frames']=sorted(required)
    rows[90]=row(90,25,'uncertain',source='manual',reviewed=True)
    assert required_quality_frames(rows,plan)==required|{89,90,91}
