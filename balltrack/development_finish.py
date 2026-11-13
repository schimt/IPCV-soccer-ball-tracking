"""Resume the declared NN validation protocol after the expanded training run finishes.

Uses only SoccerNet caches and game-9 images. It never runs the final test.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from .common import config, digest, save_json


ROOT = Path('results/development_v2')
SEQUENCES = ['SNMOT-151','SNMOT-156','SNMOT-161','SNMOT-166']


def run(*args):
    print('Running:', ' '.join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    record_path = Path('runs/nn_expanded_v2/training_record.json')
    previous = None
    while True:
        record = json.loads(record_path.read_text())
        if record['status'] == 'completed':
            break
        if not args.wait:
            raise RuntimeError('Training has not completed. Use --wait to continue automatically.')
        csv_path = record_path.parent/'fit/results.csv'
        epochs = len(list(csv.DictReader(csv_path.open()))) if csv_path.exists() else 0
        if epochs != previous:
            previous = epochs
            save_json(ROOT/'progress.json',dict(stage='waiting_for_training',completed_epochs=epochs,
                      maximum_epochs=record['settings']['epochs'],training_record=str(record_path)))
            print(f'Waiting: {epochs}/{record["settings"]["epochs"]} epochs completed',flush=True)
        time.sleep(30)
    weights = Path(record['best_weights'])
    expected_hash = digest(weights)
    save_json(ROOT/'progress.json',dict(stage='caching_expanded_NN',weights_sha256=expected_hash))
    caches = [str(ROOT/'baseline_nn_cache.json')]
    for size in [960,1280]:
        cfg = config('configs/development_v2_train.yaml')
        cfg['nn'].update(weights=str(weights),imgsz=size,confidence=.001)
        cfg_path = Path(f'configs/development_v2_nn_{size}.yaml')
        cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        cache_path = ROOT/f'expanded_nn_{size}_cache.json'
        if not cache_path.exists():
            run('-m','balltrack.development','cache','--root','data/soccernet_v2/train',
                '--sequences',*SEQUENCES,'--method','nn','--weights',str(weights),
                '--config',str(cfg_path),'--output',str(cache_path))
        cached=json.loads(cache_path.read_text())
        if cached['model']['weights_sha256'] != expected_hash or cached['config']['nn']['imgsz'] != size:
            raise ValueError('Existing cache does not match the completed training checkpoint')
        caches.append(str(cache_path))
        diagnostic=ROOT/f'expanded_nn_{size}_diagnostics.json'
        if not diagnostic.exists():
            with (ROOT/f'expanded_nn_{size}_diagnostics.log').open('w') as log:
                subprocess.run([sys.executable,'-m','balltrack.development','diagnose','--cache',str(cache_path),
                                '--output',str(diagnostic)],stdout=log,check=True)
    if digest(weights) != expected_hash:
        raise ValueError('Weights changed during validation')
    selection=ROOT/'nn_final_selection'
    save_json(ROOT/'progress.json',dict(stage='selecting_NN',weights_sha256=expected_hash))
    if not (selection/'selection.json').exists():
        run('-m','balltrack.development','select','--caches',*caches,'--output',str(selection))
    for method in ['classical','nn']:
        selected=ROOT/f'{method}_final_selection'
        shutil.copy2(selected/'selected.yaml',Path('configs')/f'development_v2_{method}.yaml')
    if not (ROOT/'nn_videos/export.json').exists():
        run('-m','balltrack.development','export','--selection',str(selection),
            '--baseline',str(ROOT/'baseline_nn_cache.json'),'--output',str(ROOT/'nn_videos'))
    save_json(ROOT/'progress.json',dict(stage='validation_completed',weights_sha256=expected_hash,
              selection=str(selection/'selection.json'),final_test_run=False))
    print('Expanded validation completed. Review diagnostic images and update report.',flush=True)


if __name__ == '__main__':
    main()
