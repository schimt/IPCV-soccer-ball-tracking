Test clip 2 (the course's Denmark-Wales video on Moodle, television broadcast). The video is not included in the repository.

Place it here as `denmark_wales_clip_1min.mp4` (640x360, 25 fps, 1500 frames). The expected SHA256 is recorded in `configs/two_clip_experiment.json`.

- `annotations.csv`: reference annotation, one row per frame
- `annotations.audit.jsonl`, `annotations.meta.json`: change log and metadata
- `quality_plan.json`, `quality_review.md`: the quality-review queue (status boundaries, first/last frame, seeded 10% sample of visible frames)
- `quality_priority.json`: supplementary review list used by `Review-Test2Quality.ps1 -Priority`
