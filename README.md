# IPCV - soccer ball tracking

Mini-project in Image Processing and Computer Vision, Aalborg University, November 2026.

## Contents

| Path | Contents |
| --- | --- |
| `balltrack/` | Source code: data download and conversion, detectors, tracker, training, annotation tool, evaluation |
| `configs/` | Configurations for training, parameter selection and the evaluated versions |
| `data/test/`, `data/test_2/` | Annotations for test clip 1 and test clip 2 |
| `data/soccernet*/`, `data/yolo*/` | Lists of the SoccerNet sequences and frames used for training and validation |
| `runs/` | Per-frame predictions on both clips, frozen evaluated code and training settings |
| `results/` | Final result tables for both clips and SoccerNet validation metrics |
| `docs/` | Assignment text, evaluation protocols and annotation guides |
| `tests/` | Unit tests |
| `Annotate-*.ps1`, `Review-Test2Quality.ps1` | Scripts for starting the annotation tool |

The stored results can be recomputed without videos or weights:

```powershell
python -m balltrack.reproduce score --check
```

## License

AGPL-3.0, see `LICENSE`.
