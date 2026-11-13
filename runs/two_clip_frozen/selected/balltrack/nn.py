"""YOLOv8n transfer learning. NN imports are lazy; classical never imports torch."""
from pathlib import Path
import json
import os
import time

import yaml

from .common import config, digest, environment, save_json
from .tracking import Candidate


def setup():
    Path('.ultralytics').mkdir(exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(Path('.ultralytics').resolve()))
    from ultralytics import YOLO
    return YOLO


def device_check(device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    if str(device) != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Set nn.device to cpu; do not assume nvidia-smi proves compatibility.')
        index = int(str(device).removeprefix('cuda:'))
        x = torch.ones((16, 16), device=f'cuda:{index}')
        (x @ x).sum().item()
    return dict(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                device=str(device), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


class NeuralDetector:
    def __init__(self, cfg, weights=None, pretrained=False):
        self.cfg = cfg
        chosen = weights or cfg['weights']
        if not pretrained and not Path(chosen).is_file():
            raise ValueError('Provide trained --weights, or explicitly use --pretrained for a COCO-only baseline')
        self.model = setup()(chosen)
        expected = 'sports ball' if pretrained else 'ball'
        ids = [i for i, name in self.model.names.items() if name.lower() == expected]
        if len(ids) != 1 or (not pretrained and len(self.model.names) != 1):
            raise ValueError(f'Expected {expected!r} class metadata; check weights/pretrained flag')
        self.class_id = ids[0]
        self.metadata = dict(weights=str(chosen), weights_sha256=digest(chosen),
                             training_status='COCO pretrained only' if pretrained else 'custom ball weights (see training run)',
                             **device_check(cfg['device']))

    def detect(self, frame):
        result = self.model.predict(frame, imgsz=self.cfg['imgsz'], conf=self.cfg['confidence'],
                    device=self.cfg['device'], classes=[self.class_id], verbose=False)[0]
        candidates = []
        for coords, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x, y, x2, y2 = map(float, coords)
            candidates.append(Candidate((x, y, x2-x, y2-y), float(score)))
        return candidates


def train(args):
    cfg = config(args.config)
    data = Path(args.data).resolve()
    manifest_path = data.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    train_games = {s['game'] for s in manifest['sequences'] if s['split'] == 'train'}
    val_games = {s['game'] for s in manifest['sequences'] if s['split'] == 'val'}
    if train_games & val_games or not train_games or not val_games:
        raise ValueError('Invalid game-level training/validation split')
    spec = yaml.safe_load(data.read_text(encoding='utf-8'))
    if 'test' in spec or 'download' in spec:
        raise ValueError('Training YAML must contain only local train/val data')
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new training run directory')
    settings = dict(cfg['train'])
    if args.epochs is not None:
        settings['epochs'] = args.epochs
    if args.imgsz is not None:
        settings['imgsz'] = args.imgsz
    if args.batch is not None:
        settings['batch'] = args.batch
    device = args.device or cfg['nn']['device']
    machine = device_check(device)
    model = setup()(args.weights or cfg['nn']['weights'])
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    record = dict(status='started', purpose=args.purpose, settings=settings, seed=cfg['seed'],
                  data=str(data), data_manifest_sha256=digest(manifest_path),
                  data_manifest=manifest, environment=environment(), hardware=machine,
                  initialization=args.weights or cfg['nn']['weights'])
    save_json(out / 'training_record.json', record)
    model.train(data=str(data), device=device, seed=cfg['seed'], deterministic=True,
                project=str(out), name='fit', exist_ok=True, plots=True, **settings)
    record.update(status='completed', elapsed_seconds=time.perf_counter()-started,
                  best_weights=str(out / 'fit/weights/best.pt'))
    save_json(out / 'training_record.json', record)
    print('Training record:', out / 'training_record.json')


def parser(sub):
    p = sub.add_parser('train')
    p.add_argument('--data', default='data/yolo/dataset.yaml')
    p.add_argument('--output', default='runs/train_ball')
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--weights')
    p.add_argument('--device')
    p.add_argument('--epochs', type=int)
    p.add_argument('--imgsz', type=int)
    p.add_argument('--batch', type=int)
    p.add_argument('--purpose', choices=['development', 'smoke-test'], default='development')
    p.set_defaults(func=train)
