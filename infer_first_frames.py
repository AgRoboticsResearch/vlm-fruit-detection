#!/usr/bin/env python
"""YOLO11-seg inference on the first frame of every episode in a lerobot v3.0 dataset.

Episodes are timestamp ranges inside shared video files (meta/episodes/*.parquet);
the first frame of episode i is at from_timestamp in its video file.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

CAM = 'observation.images.camera'
COL = f'videos/{CAM}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', default='weights/yolov11-m-best.pt')
    p.add_argument('--dataset', required=True, help='lerobot dataset root')
    p.add_argument('--out', required=True, help='output directory for JPGs')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--device', default='0')
    args = p.parse_args()

    ds = Path(args.dataset)
    eps_file = next((ds / 'meta' / 'episodes').glob('chunk-*/*.parquet'))
    eps = pd.read_parquet(eps_file)
    fcol, tcol = f'{COL}/file_index', f'{COL}/from_timestamp'
    ccol = f'{COL}/chunk_index'

    info = json.loads((ds / 'meta' / 'info.json').read_text())
    h, w = info['features'][CAM]['shape'][:2]
    nbytes = w * h * 3

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    n_dets = 0
    for i, row in eps.iterrows():
        vid = ds / 'videos' / CAM / f"chunk-{int(row[ccol]):03d}" / f"file-{int(row[fcol]):03d}.mp4"
        t = row[tcol]
        r = subprocess.run(
            ['ffmpeg', '-v', 'error', '-ss', f'{t:.6f}', '-i', str(vid),
             '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
            capture_output=True, check=True)
        assert len(r.stdout) == nbytes, f'{vid.name} t={t}: got {len(r.stdout)} bytes'
        frame = np.frombuffer(r.stdout, np.uint8).reshape(h, w, 3)
        res = model.predict(frame, conf=args.conf, device=args.device, verbose=False)[0]
        cv2.imwrite(str(out / f'episode_{int(row["episode_index"]):06d}_first.jpg'), res.plot())
        n_dets += len(res.boxes)
        if (i + 1) % 100 == 0 or i == len(eps) - 1:
            print(f'[{i + 1}/{len(eps)}] episodes done, {n_dets} dets so far', flush=True)


if __name__ == '__main__':
    main()
