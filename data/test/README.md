Test clip 1 (own clip, television broadcast). The video is not included in the repository; it was submitted separately on Moodle.

Place it here as `clip.mp4` (1350x1080, 25 fps, 1500 frames). The expected SHA256 is recorded in `configs/final_test.json`.

- `annotations.csv`: reference annotation, one row per frame
- `annotations.audit.jsonl`, `annotations.meta.json`: change log and metadata
- `annotation_draft.json`: unreviewed keyframe and interval proposals used before the manual review
- `consistency_review.json`: record of the 31-frame consistency re-check
- `clip_manifest.json`: how the clip was cut from the source video
