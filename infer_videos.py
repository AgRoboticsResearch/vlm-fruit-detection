#!/usr/bin/env python
"""Minimal YOLO11-seg inference: annotate every video in a directory.

Frames are decoded through an ffmpeg pipe (sources are AV1, which the local
OpenCV build cannot decode) and annotated frames are written as mp4v.
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return w, h, fps


def ffmpeg_frames(path, size):
    """Yield BGR frames decoded by system ffmpeg (handles AV1)."""
    w, h = size
    proc = subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-i', str(path),
         '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
        stdout=subprocess.PIPE)
    frame_bytes = w * h * 3
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.stdout.close()
    proc.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', default='weights/yolov11-m-best.pt')
    p.add_argument('--source', required=True, help='directory containing videos')
    p.add_argument('--out', default='runs/inference', help='output directory')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--device', default='0')
    args = p.parse_args()

    videos = sorted(Path(args.source).glob('*.mp4'))
    assert videos, f'no .mp4 files in {args.source}'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    for i, vid in enumerate(videos, 1):
        w, h, fps = video_meta(vid)
        writer = cv2.VideoWriter(str(out / f'{vid.stem}.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        n = 0
        for frame in ffmpeg_frames(vid, (w, h)):
            r = model.predict(frame, conf=args.conf, device=args.device,
                              verbose=False)[0]
            writer.write(r.plot())
            n += 1
        writer.release()
        print(f'[{i}/{len(videos)}] {vid.name}: {n} frames -> {out / (vid.stem + ".mp4")}', flush=True)


if __name__ == '__main__':
    main()
