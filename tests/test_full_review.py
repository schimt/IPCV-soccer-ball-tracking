import json
from types import SimpleNamespace

import cv2
import pytest

from balltrack import annotate
from balltrack.common import digest, read_rows, row, save_json, write_rows
from test_sparse_review import mock_gui, video


def test_expansion_preserves_completed_rows_and_restores_only_pending_proposals():
    current = [row(i, 25, 'visible', [10, 10, 5, 5], source='manual_review', reviewed=True)
               for i in range(0, 1500, 5)]
    proposals = [row(i, 25, 'visible', [20, 20, 4, 4], source='interpolated_linear', reviewed=True)
                 for i in range(1500)]
    expanded = annotate.expand_review_rows(current, proposals, 1500, 25)
    assert [r['frame'] for r in expanded] == list(range(1500))
    assert expanded[::5] == current
    assert all(r['reviewed'] == 0 and r['x'] == 20 for r in expanded if r['frame'] % 5)
    assert annotate.annotation_progress(expanded)['reviewed'] == 300
    assert annotate.annotation_progress(expanded)['unreviewed'] == 1200
    assert annotate.first_pending(expanded) == 1
    # New rows and returned rows must not mutate either input collection.
    expanded[0]['reviewed'] = 0
    assert current[0]['reviewed'] == 1 and proposals[1]['reviewed'] == 1


def test_expansion_without_proposals_leaves_missing_frames_unlabeled():
    existing = [row(0, 25, 'out_of_frame', source='manual', reviewed=True)]
    expanded = annotate.expand_review_rows(existing, [], 3, 25)
    assert expanded[0] == existing[0]
    assert [r['status'] for r in expanded] == ['out_of_frame', 'unlabeled', 'unlabeled']
    assert expanded[2]['timestamp'] == .08


def setup_pending_gui(tmp_path):
    clip, csv = tmp_path/'clip.mp4', tmp_path/'annotations.csv'
    video(clip)
    rows, _ = annotate.initialize(clip, csv, stride=1)
    rows = [row(i, 25, 'out_of_frame', source='manual', reviewed=(i % 5 == 0)) for i in range(15)]
    write_rows(csv, rows)
    return csv, read_rows(csv, 'gt'), SimpleNamespace(video=str(clip), output=str(csv),
                                                    stride=1, pending_only=True, init_only=False)


def capture_displayed_frames(monkeypatch):
    seen = []
    def put_text(image, text, *args):
        if text.startswith('Frame '):
            seen.append(int(text.split()[1].split('/')[0]))
        return image
    monkeypatch.setattr(cv2, 'putText', put_text)
    return seen


def test_pending_gui_skips_all_approved_frames_and_closes_when_complete(tmp_path, monkeypatch):
    csv, original, args = setup_pending_gui(tmp_path)
    mock_gui(monkeypatch, 'rouf'*3)
    displayed = capture_displayed_frames(monkeypatch)
    annotate.main(args)
    assert displayed == [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]
    result = read_rows(csv, 'gt')
    assert result[::5] == original[::5]
    assert annotate.annotation_progress(result)['reviewed'] == 15
    audit = [json.loads(line) for line in csv.with_suffix('.audit.jsonl').read_text().splitlines()]
    assert [r['frame'] for r in audit] == displayed
    monkeypatch.setattr(cv2, 'namedWindow', lambda *a: pytest.fail('Completed review must not open a window'))
    annotate.main(args)


def test_pending_gui_navigation_and_resume_never_show_approved_rows(tmp_path, monkeypatch):
    csv, original, args = setup_pending_gui(tmp_path)
    mock_gui(monkeypatch, 'ddddanpq')
    displayed = capture_displayed_frames(monkeypatch)
    annotate.main(args)
    assert displayed == [1, 2, 3, 4, 6, 4, 6, 7]
    assert read_rows(csv, 'gt') == original  # navigation/playback approve nothing
    mock_gui(monkeypatch, 'oq')
    displayed.clear()
    annotate.main(args)
    assert displayed == [1, 2]
    mock_gui(monkeypatch, 'q')
    displayed.clear()
    annotate.main(args)
    assert displayed == [2]
    assert read_rows(csv, 'gt')[::5] == original[::5]


def test_pending_wrap_recovers_skipped_frame_and_empty_queue_returns_none():
    rows = [row(i, 25, 'out_of_frame', source='manual', reviewed=i != 1) for i in range(6)]
    assert annotate.pending_neighbor(rows, 5, wrap=True) == 1
    assert annotate.pending_neighbor(rows, 1, direction=-1) == 1
    rows[1]['reviewed'] = 1
    assert annotate.pending_neighbor(rows, 5, wrap=True) is None


def test_closing_stale_gui_cannot_restore_old_csv(tmp_path, monkeypatch):
    csv, original, args = setup_pending_gui(tmp_path)
    mock_gui(monkeypatch, 'q')
    newer = [dict(r) for r in original]
    newer[1] = row(1, 25, 'uncertain', source='manual', reviewed=True)
    expected_hash = []
    def external_change_then_quit(delay):
        write_rows(csv, newer)
        expected_hash.append(digest(csv))
        return ord('q')
    monkeypatch.setattr(cv2, 'waitKey', external_change_then_quit)
    with pytest.raises(RuntimeError, match='not overwritten'):
        annotate.main(args)
    assert digest(csv) == expected_hash[0]
    result = read_rows(csv, 'gt')
    assert result[1]['status'] == 'uncertain' and result[1]['reviewed'] == 1
    assert result[:1] + result[2:] == original[:1] + original[2:]
    recovered = list((tmp_path/'backups').glob('*_session_conflict/annotations.csv'))
    assert len(recovered) == 1 and read_rows(recovered[0], 'gt') == original


@pytest.mark.parametrize('changed_file', ['csv', 'metadata'])
def test_stale_edit_is_recovered_without_overwriting_external_changes(tmp_path, changed_file):
    csv, rows, _ = setup_pending_gui(tmp_path)
    session = annotate.AnnotationSession(csv)
    rows[1] = row(1, 25, 'visible', [10, 10, 5, 5], source='manual', reviewed=True)
    if changed_file == 'csv':
        external = read_rows(csv, 'gt')
        external[2] = row(2, 25, 'uncertain', source='manual', reviewed=True)
        write_rows(csv, external)
    else:
        metadata = json.loads(csv.with_suffix('.meta.json').read_text())
        metadata['annotation_stride'] = 5
        save_json(csv.with_suffix('.meta.json'), metadata)
    expected_hash = digest(csv)
    expected_meta = csv.with_suffix('.meta.json').read_bytes()
    with pytest.raises(RuntimeError, match='not overwritten'):
        session.save(rows)
    assert digest(csv) == expected_hash
    assert csv.with_suffix('.meta.json').read_bytes() == expected_meta
    recovered = list((tmp_path/'backups').glob('*_session_conflict/annotations.csv'))
    assert read_rows(recovered[0], 'gt')[1]['status'] == 'visible'
    assert session.conflicted


def test_session_can_save_several_edits_and_its_own_metadata_change(tmp_path):
    csv, rows, _ = setup_pending_gui(tmp_path)
    session = annotate.AnnotationSession(csv)
    rows[1]['reviewed'] = 1
    session.save(rows)
    metadata = json.loads(csv.with_suffix('.meta.json').read_text())
    metadata['interpolation'] = True
    session.save(rows, metadata=metadata)
    rows[2]['reviewed'] = 1
    session.save(rows)
    session.check(rows)
    assert read_rows(csv, 'gt')[2]['reviewed'] == 1


def test_readonly_quit_does_not_rewrite_annotations(tmp_path, monkeypatch):
    csv, _, args = setup_pending_gui(tmp_path)
    mock_gui(monkeypatch, 'q')
    original_mtime = csv.stat().st_mtime_ns
    annotate.main(args)
    assert csv.stat().st_mtime_ns == original_mtime


def test_selected_review_saves_full_csv_and_leaves_other_rows_unchanged(tmp_path, monkeypatch):
    csv, original, args = setup_pending_gui(tmp_path)
    selection = tmp_path/'selection.md'
    selection.write_text('| Frame | Time |\n|---|---|\n| 2 | 0.08 |\n| 7 | 0.28 |\n')
    args.frames_file = str(selection)
    mock_gui(monkeypatch, 'uf')
    displayed = capture_displayed_frames(monkeypatch)
    annotate.main(args)
    assert displayed == [2, 7]
    result = read_rows(csv, 'gt')
    assert len(result) == 15
    assert result[2]['status'] == 'uncertain' and result[2]['reviewed'] == 1
    assert result[7]['reviewed'] == 1
    assert all(result[i] == original[i] for i in range(15) if i not in {2, 7})


def test_selected_review_resumes_without_resetting_approvals(tmp_path, monkeypatch):
    csv, _, args = setup_pending_gui(tmp_path)
    selection = tmp_path/'selection.md'
    selection.write_text('| 2 |\n| 7 |\n')
    args.frames_file = str(selection)
    mock_gui(monkeypatch, 'uq')
    annotate.main(args)
    mock_gui(monkeypatch, 'fq')
    displayed = capture_displayed_frames(monkeypatch)
    annotate.main(args)
    assert displayed == [7]
