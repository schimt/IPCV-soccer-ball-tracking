# Soccer ball tracking: classical pipeline vs. YOLOv8n

Mini-project in Image Processing and Computer Vision, Aalborg University, autumn 2026.

The project compares two ways of tracking the ball in soccer video: a classical OpenCV pipeline (morphological top-hat filtering and blob filtering) and a YOLOv8n detector fine-tuned on SoccerNet tracking data. Both detectors use the same Kalman tracker with camera-motion compensation. The methods were trained and tuned on SoccerNet only, and evaluated on two manually annotated one-minute test clips: an own broadcast clip and the course's Denmark-Wales clip.

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

The report, the test videos and the trained model weights are not included. The report and the own test clip are submitted on Moodle.

The stored results can be recomputed without videos or weights:

```powershell
python -m balltrack.reproduce score --check
```

## License

AGPL-3.0, see `LICENSE`.
