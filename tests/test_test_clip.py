import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from balltrack.annotate import annotation_progress, import_draft, initialize, interpolate_between
from balltrack.common import digest, read_rows, row, save_json
from balltrack.test_clip import extract_interval


def make_video(path, count=15, fps=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (80, 64))
    assert writer.isOpened()
    for i in range(count):
        writer.write(np.full((64, 80, 3), 15*i, np.uint8))
    writer.release()


def test_exact_interval_and_frame_mapping(tmp_path):
    source, target = tmp_path/'source.mp4', tmp_path/'clip.mp4'
    make_video(source)
    manifest = extract_interval(source, target, .4, duration=1)
    assert manifest['source_start_frame'] == 2
    assert manifest['source_end_frame_exclusive'] == 7
    assert manifest['decoded_output_frames'] == 5
    assert manifest['clip']['duration_seconds'] == 1
    assert manifest['last_frame_timestamp'] == pytest.approx(.8)
    original = cv2.VideoCapture(str(source)); original.set(cv2.CAP_PROP_POS_FRAMES, 2)
    _, expected = original.read(); original.release()
    extracted = cv2.VideoCapture(str(target)); _, first = extracted.read(); extracted.release()
    assert abs(float(expected.mean())-float(first.mean())) < 5  # lossy video re-encoding tolerance
    old_hash = digest(target)
    with pytest.raises(ValueError, match='replace'):
        extract_interval(source, target, 0, duration=1)
    assert digest(target) == old_hash


def test_invalid_intervals_never_create_clip(tmp_path):
    source, target = tmp_path/'source.mp4', tmp_path/'clip.mp4'
    make_video(source)
    with pytest.raises(ValueError, match='align'):
        extract_interval(source, target, .11, duration=1)
    with pytest.raises(ValueError, match='exceeds'):
        extract_interval(source, target, 0, duration=60)
    assert not target.exists()


def test_interpolation_never_marks_reviewed_and_preserves_endpoints():
    rows = [row(i, 25, 'unlabeled') for i in range(5)]
    rows[0] = row(0, 25, 'visible', [10, 10, 4, 4], source='manual', reviewed=True)
    rows[4] = row(4, 25, 'visible', [18, 18, 8, 8], source='manual', reviewed=True)
    interpolate_between(rows, 0, 4, 25)
    assert rows[2]['x'] == 14 and rows[2]['w'] == 6
    assert all(r['source'] == 'interpolated_linear' and not r['reviewed'] for r in rows[1:4])
    assert rows[0]['reviewed'] == rows[4]['reviewed'] == 1
    assert annotation_progress(rows)['reviewed'] == 2


def test_interpolation_rejects_occlusion_or_existing_labels():
    rows = [row(0, 25, 'visible', [1, 1, 5, 5]), row(1, 25, 'occluded'),
            row(2, 25, 'visible', [5, 5, 5, 5])]
    with pytest.raises(ValueError, match='visibility'):
        interpolate_between(rows, 0, 2, 25)
    assert rows[1]['status'] == 'occluded'


def test_draft_import_is_unreviewed_and_preserves_existing_work(tmp_path):
    source, annotations = tmp_path/'source.mp4', tmp_path/'annotations.csv'
    make_video(source)
    _, meta = initialize(source, annotations)
    draft = tmp_path/'draft.json'
    spec = dict(video_sha256=meta['sha256'],
                anchors=[dict(frame=0, status='visible', box=[10, 10, 4, 4]),
                         dict(frame=4, status='visible', box=[18, 18, 4, 4])], interpolate=[[0, 4]])
    save_json(draft, spec)
    args = SimpleNamespace(annotations=annotations, draft=draft)
    import_draft(args)
    rows = read_rows(annotations, 'gt')
    assert annotation_progress(rows)['reviewed'] == 0
    assert rows[2]['source'] == 'interpolated_linear'
    before = digest(annotations)
    with pytest.raises(ValueError, match='untouched'):
        import_draft(args)
    assert digest(annotations) == before


def test_registered_test_rejects_other_video(tmp_path, monkeypatch):
    import balltrack.test_clip as module
    one, two, registry = tmp_path/'one.mp4', tmp_path/'two.mp4', tmp_path/'registry.json'
    one.write_bytes(b'video-one'); two.write_bytes(b'video-two')
    save_json(registry, dict(clip=dict(sha256=digest(one))))
    monkeypatch.setattr(module, 'REGISTRY', registry)
    assert module.validate_registered_video(one)['clip']['sha256'] == digest(one)
    with pytest.raises(ValueError, match='registered'):
        module.validate_registered_video(two)
