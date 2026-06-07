# -*- coding: utf-8 -*-
"""검출 분기 진단: 식판 등 다중 음식 사진이 왜 single 로 분기됐는지 본다.

- checkpoints/yolo11n_23cls.pt 로 conf=0.05 까지 낮춰 추론 → 검출된 박스 전부 출력
- conf=0.35(파이프라인 기본) 기준 생존 박스 수 + 분기 판정(0~1→single, ≥2→multi) 보고
- 박스 그린 시각화를 outputs/ 에 저장(초록=생존 conf≥0.35, 회색=0.05~0.35 미달)

사용:
    python scripts/diagnose_detect.py "C:\\path\\to\\sikpan.jpg"
    python scripts/diagnose_detect.py "<path>" --low 0.05 --pipe 0.35
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import cv2
from ultralytics import YOLO

WEIGHTS = "checkpoints/yolo11n_23cls.pt"
OUT_DIR = "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="진단할 사진 경로")
    ap.add_argument("--low", type=float, default=0.05, help="낮춘 검출 임계값(기본 0.05)")
    ap.add_argument("--pipe", type=float, default=0.35, help="파이프라인 기준 임계값(기본 0.35)")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not os.path.isfile(WEIGHTS):
        print(f"[오류] 가중치 없음: {WEIGHTS}", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(args.image):
        print(f"[오류] 이미지 없음: {args.image}", file=sys.stderr); sys.exit(1)

    model = YOLO(WEIGHTS)
    names = model.names
    res = model.predict(source=args.image, conf=args.low, iou=0.45,
                        imgsz=args.imgsz, device="cpu", verbose=False)[0]

    img = res.orig_img.copy()  # BGR
    H, W = img.shape[:2]
    boxes = []
    for b in res.boxes:
        conf = float(b.conf.item())
        cid = int(b.cls.item())
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        boxes.append((conf, cid, names.get(cid, str(cid)), (x1, y1, x2, y2)))
    boxes.sort(key=lambda t: t[0], reverse=True)

    print(f"\n이미지: {args.image}  (W={W} H={H})")
    print(f"검출(conf≥{args.low}) 총 {len(boxes)}개 — 클래스 / conf / xyxy:")
    print("-" * 64)
    for i, (conf, cid, name, (x1, y1, x2, y2)) in enumerate(boxes, 1):
        mark = "✓생존" if conf >= args.pipe else " 미달"
        print(f"{i:2}. [{mark}] {name:16} conf={conf:.3f}  "
              f"xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    survivors = [b for b in boxes if b[0] >= args.pipe]
    n = len(survivors)
    branch = "multi" if n >= 2 else "single"
    print("-" * 64)
    print(f"conf≥{args.pipe} 생존 박스: {n}개  →  파이프라인 분기: {branch.upper()}"
          + (f"  (0~1개이므로 전체 이미지 단일 분류)" if n < 2 else ""))

    # 시각화: 생존=초록, 미달=회색
    for conf, cid, name, (x1, y1, x2, y2) in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        color = (0, 200, 0) if conf >= args.pipe else (160, 160, 160)
        cv2.rectangle(img, p1, p2, color, 2)
        cv2.putText(img, f"{name} {conf:.2f}", (p1[0], max(12, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out = os.path.join(OUT_DIR, f"diag_{stem}.jpg")
    cv2.imwrite(out, img)
    print(f"시각화 저장: {out}  (초록=생존 conf≥{args.pipe}, 회색=미달)")


if __name__ == "__main__":
    main()
