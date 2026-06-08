"""YOLO 23종 객체탐지 데이터셋 빌더 (Phase 1 검출기 — 다중 클래스 버전).

aihub_extracted/<한글클래스>/ 의 라벨을 읽어 YOLO 표준 포맷으로 변환·분할한다.

라벨 출처(클래스 폴더 안):
    - kfood 18종: crop_area.properties  →  '파일명=x,y,w,h'  (좌상단 x,y + 너비/높이)
        ※ 'VOC x1,y1,x2,y2' 가 아니라 'x,y,w,h' 임을 원본 이미지 크기로 검증함.
          (XYWH 해석 시 박스가 항상 이미지 범위 안, VOC 해석 시 다수가 범위 초과)
    - UEC 5종(돈까스·밥·소시지·스파게티·햄버거): bb_info.txt  →  '<id> x1 y1 x2 y2'  (VOC)
        ※ 헤더 'img x1 y1 x2 y2' 존재. 한 id 가 여러 줄이면 박스 여러 개.

공통 변환:
    - PIL 로 원본 (W,H) 획득 → YOLO 정규화(class xc yc w h, 0~1).
    - 박스 픽셀좌표를 [0,W]/[0,H] 로 clip 후 w<=0 또는 h<=0 이면 skip.
    - 정규화 좌표 다시 [0,1] clip.
    - 한 이미지에 박스 여러 개면 모두 라벨에 포함.
    - 라벨(유효 박스 >=1)이 있는 이미지만 데이터셋에 포함.

출력(YOLO 표준):
    data/detector/images/{train,val,test}/<english>__<stem><ext>
    data/detector/labels/{train,val,test}/<english>__<stem>.txt
    (파일명 충돌 방지를 위해 클래스 영문명 접두어 '<english>__' 부여)

클래스 인덱스: 영문 23개 알파벳순(bibimbap=0 … tteokbokki=22).
    checkpoints/best_efficientnet_b0_23.pt 의 classes 와 동일.

분할: 클래스별 8:1:1 (train/val/test) 층화, seed 42.
"""

import argparse
import json
import os
import shutil
import sys
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "val", "test")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_SOURCE = r"C:\Users\yunwo\Documents\aihub_extracted"
_DEFAULT_TARGET = os.path.join(_PROJECT_ROOT, "data", "detector")
_DEFAULT_YAML = os.path.join(_PROJECT_ROOT, "configs", "detector_data.yaml")
_FOOD_TABLE = os.path.join(_PROJECT_ROOT, "food_calories.json")
# 수동 라벨(테스트 전용) 보존 위치: labels_manual/<english>/crop_area_manual.properties
# 자동 8:1:1 분할에서 제외하고 항상 test 로만 병합한다(train/val 불변, 재빌드에도 유지).
_MANUAL_ROOT = os.path.join(_PROJECT_ROOT, "labels_manual")

# 영문 클래스명 → 소스 폴더명(한글, NFC). 분류기 prepare_data 와 동일한 매핑 규칙.
ENGLISH_TO_FOLDER: Dict[str, str] = {
    "bibimbap": "비빔밥",
    "bulgogi": "불고기",
    "doenjang-jjigae": "된장찌개",
    "donkkaseu": "돈까스",       # UEC
    "fried-egg": "계란후라이",
    "galbitang": "갈비탕",
    "gimbap": "김밥",
    "hamburger": "햄버거",       # UEC
    "japchae": "잡채",
    "jeyuk-bokkeum": "제육볶음",
    "jjajangmyeon": "짜장면",
    "jjamppong": "짬뽕",
    "kalguksu": "칼국수",
    "kimchi-jjigae": "김치찌개",
    "mul-naengmyeon": "물냉면",
    "pizza": "피자",
    "ramyeon": "라면",
    "rice": "밥",               # UEC
    "samgyeopsal": "삼겹살",
    "sausage": "소시지",         # UEC
    "spaghetti": "스파게티",     # UEC
    "sundubu-jjigae": "순두부찌개",
    "tteokbokki": "떡볶이",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="aihub_extracted → YOLO 23종 검출 데이터셋")
    p.add_argument("--source", default=_DEFAULT_SOURCE, help="aihub_extracted 경로")
    p.add_argument("--target", default=_DEFAULT_TARGET, help="data/detector 경로")
    p.add_argument("--yaml-out", default=_DEFAULT_YAML, help="configs/detector_data.yaml 경로")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t)


def class_order() -> List[str]:
    """food_calories.json 의 23개 english 를 알파벳순으로 반환(bibimbap=0 … tteokbokki=22)."""
    with open(_FOOD_TABLE, encoding="utf-8") as f:
        table = json.load(f)
    names = sorted(info["english"] for info in table)
    if names != sorted(ENGLISH_TO_FOLDER):
        print("[중단] food_calories.json english ↔ 매핑 불일치", file=sys.stderr)
        print("  json:", names, file=sys.stderr)
        print("  map :", sorted(ENGLISH_TO_FOLDER), file=sys.stderr)
        sys.exit(1)
    return names


def find_image(class_dir: str, stem: str) -> Optional[str]:
    """stem 에 대응하는 이미지 파일 경로(대소문자/확장자 변형 허용). 없으면 None."""
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        p = os.path.join(class_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def parse_crop_area(path: str) -> Tuple[Dict[str, List[Tuple[int, int, int, int]]], int]:
    """crop_area.properties → {stem: [(x1,y1,x2,y2), ...]}, malformed 줄 수.

    원본 형식은 'stem=x,y,w,h' (좌상단+너비/높이). 여기서 (x1,y1,x2,y2)=(x,y,x+w,y+h) 로 변환해 반환.
    값이 4개 정수가 아니면(예: '=dissimilar,..', '=,986,590') malformed 로 집계.
    """
    boxes: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    malformed = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            parts = val.split(",")
            if len(parts) != 4:
                malformed += 1
                continue
            try:
                x, y, w, h = (int(v) for v in parts)
            except ValueError:
                malformed += 1
                continue
            boxes[key.strip()].append((x, y, x + w, y + h))
    return boxes, malformed


def load_manual(english: str) -> Tuple[Dict[str, List[Tuple[int, int, int, int]]], int]:
    """labels_manual/<english>/crop_area_manual.properties (있으면) → 박스 dict.

    수동 라벨은 AI Hub crop_area 와 동일한 'stem=x,y,w,h' 포맷으로 보존한다.
    이 stem 들은 자동 8:1:1 분할에서 제외되고 항상 test 로만 병합된다(train/val 불변).
    """
    path = os.path.join(_MANUAL_ROOT, english, "crop_area_manual.properties")
    if not os.path.isfile(path):
        return {}, 0
    return parse_crop_area(path)


def parse_bb_info(path: str) -> Tuple[Dict[str, List[Tuple[int, int, int, int]]], int]:
    """bb_info.txt → {id: [(x1,y1,x2,y2), ...]}, malformed 줄 수. (VOC, 헤더/멀티박스 처리)"""
    boxes: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    malformed = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            parts = line.split()
            if len(parts) != 5:
                if line.strip():
                    malformed += 1
                continue
            if i == 0 and parts[0].lower() == "img":  # 헤더
                continue
            try:
                img_id = str(int(parts[0]))
                x1, y1, x2, y2 = (int(v) for v in parts[1:])
            except ValueError:
                malformed += 1
                continue
            boxes[img_id].append((x1, y1, x2, y2))
    return boxes, malformed


def to_yolo(
    box: Tuple[int, int, int, int], W: int, H: int
) -> Optional[Tuple[float, float, float, float]]:
    """(x1,y1,x2,y2) → YOLO (xc,yc,w,h) 정규화. 픽셀 clip 후 w<=0/h<=0 이면 None."""
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    xc = (x1 + x2) / 2.0 / W
    yc = (y1 + y2) / 2.0 / H
    nw = bw / W
    nh = bh / H
    clip = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    return clip(xc), clip(yc), clip(nw), clip(nh)


def split_files(files, train_ratio, val_ratio, seed):
    """파일 목록 8:1:1 분할. 표본이 매우 적을 때 수동 처리(분류기와 동일 규칙)."""
    n = len(files)
    if n == 0:
        return [], [], []
    if n == 1:
        return files, [], []
    if n == 2:
        return files[:1], files[1:], []
    test_ratio = 1.0 - train_ratio - val_ratio
    train, rest = train_test_split(files, train_size=train_ratio, random_state=seed)
    if len(rest) <= 1:
        return train, rest, []
    rel_val = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.0
    val, test = train_test_split(rest, train_size=rel_val, random_state=seed)
    return train, val, test


def main() -> None:
    args = parse_args()
    classes = class_order()
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    if not os.path.isdir(args.source):
        print(f"[오류] source 없음: {args.source}", file=sys.stderr)
        sys.exit(1)
    actual_dirname = {nfc(d): d for d in os.listdir(args.source)
                      if os.path.isdir(os.path.join(args.source, d))}

    # 출력 폴더 새로 생성
    if os.path.isdir(args.target):
        print(f"[초기화] 기존 타겟 삭제: {args.target}")
        shutil.rmtree(args.target)
    for kind in ("images", "labels"):
        for sp in SPLITS:
            os.makedirs(os.path.join(args.target, kind, sp), exist_ok=True)

    # 집계용
    box_counts: Dict[str, int] = {}       # 클래스별 유효 박스 수
    img_counts: Dict[str, int] = {}       # 클래스별 라벨 있는 이미지 수
    split_img: Dict[str, Tuple[int, int, int]] = {}  # 클래스별 (train,val,test) 이미지 수
    skipped_boxes: Dict[str, int] = {}    # 클래스별 skip 박스(malformed + degenerate)
    no_image: Dict[str, int] = {}         # 라벨엔 있으나 이미지 파일 없음
    source_kind: Dict[str, str] = {}      # 'kfood' | 'uec'
    manual_counts: Dict[str, int] = {}    # 클래스별 수동(test 전용) 이미지 수

    for english in classes:
        idx = cls_to_idx[english]
        folder_nfc = nfc(ENGLISH_TO_FOLDER[english])
        if folder_nfc not in actual_dirname:
            print(f"[경고] 소스 폴더 없음: {english} ({folder_nfc}) — 0건 처리", file=sys.stderr)
            box_counts[english] = 0; img_counts[english] = 0
            split_img[english] = (0, 0, 0); skipped_boxes[english] = 0
            no_image[english] = 0; source_kind[english] = "-"
            manual_counts[english] = 0
            continue
        class_dir = os.path.join(args.source, actual_dirname[folder_nfc])

        bb_path = os.path.join(class_dir, "bb_info.txt")
        crop_path = os.path.join(class_dir, "crop_area.properties")
        if os.path.isfile(bb_path):
            raw_boxes, malformed = parse_bb_info(bb_path)
            source_kind[english] = "uec"
        elif os.path.isfile(crop_path):
            raw_boxes, malformed = parse_crop_area(crop_path)
            source_kind[english] = "kfood"
        else:
            print(f"[경고] 라벨 파일 없음: {english} — 0건", file=sys.stderr)
            box_counts[english] = 0; img_counts[english] = 0
            split_img[english] = (0, 0, 0); skipped_boxes[english] = 0
            no_image[english] = 0; source_kind[english] = "-"
            manual_counts[english] = 0
            continue

        # 수동(test 전용) 라벨 로드 → 자동 분할에서 제외하고 나중에 test 로만 병합
        manual_boxes, manual_malformed = load_manual(english)
        manual_stems = set(manual_boxes)

        def boxes_to_lines(boxes, W, H):
            """박스 리스트 → YOLO 라벨 라인. (degenerate 수, 라인) 반환."""
            out, sk = [], 0
            for box in boxes:
                yolo = to_yolo(box, W, H)
                if yolo is None:
                    sk += 1
                    continue
                xc, yc, nw, nh = yolo
                out.append(f"{idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            return out, sk

        # 라벨 있는 이미지별로 YOLO 라벨 라인 구성 (수동 stem 은 제외)
        skipped = malformed + manual_malformed
        missing = 0
        items: List[Tuple[str, List[str]]] = []  # (image_path, [label_lines])
        bcount = 0
        for stem, boxes in raw_boxes.items():
            if stem in manual_stems:
                continue  # 수동(test 전용) → 자동 분할에서 제외
            ip = find_image(class_dir, stem)
            if ip is None:
                missing += 1
                continue
            with Image.open(ip) as im:
                W, H = im.size
            lines, sk = boxes_to_lines(boxes, W, H)
            skipped += sk
            if lines:
                items.append((ip, lines))
                bcount += len(lines)

        # 수동(test 전용) 항목 구성 — 원본은 동일 소스 폴더(class_dir)에서 찾는다
        manual_items: List[Tuple[str, List[str]]] = []
        for stem, boxes in manual_boxes.items():
            ip = find_image(class_dir, stem)
            if ip is None:
                missing += 1
                continue
            with Image.open(ip) as im:
                W, H = im.size
            lines, sk = boxes_to_lines(boxes, W, H)
            skipped += sk
            if lines:
                manual_items.append((ip, lines))
                bcount += len(lines)
        manual_items.sort(key=lambda t: os.path.basename(t[0]))

        # 클래스별 8:1:1 분할 (이미지 단위) — 자동분할은 비수동 항목만 대상
        items.sort(key=lambda t: os.path.basename(t[0]))  # 재현성
        tr, va, te = split_files(items, args.train_ratio, args.val_ratio, args.seed)
        te = list(te) + manual_items  # 수동 라벨은 전부 test 로 병합 (train/val 불변)

        for sp, group in zip(SPLITS, (tr, va, te)):
            for ip, lines in group:
                stem = os.path.splitext(os.path.basename(ip))[0]
                ext = os.path.splitext(ip)[1].lower()
                if ext not in IMG_EXTENSIONS:
                    ext = ".jpg"
                base = f"{english}__{stem}"
                dst_img = os.path.join(args.target, "images", sp, base + ext)
                dst_lbl = os.path.join(args.target, "labels", sp, base + ".txt")
                shutil.copy2(ip, dst_img)
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")

        box_counts[english] = bcount
        img_counts[english] = len(items) + len(manual_items)
        split_img[english] = (len(tr), len(va), len(te))  # te 는 수동 병합 포함
        skipped_boxes[english] = skipped
        no_image[english] = missing
        manual_counts[english] = len(manual_items)

    # ===== detector_data.yaml =====
    os.makedirs(os.path.dirname(args.yaml_out), exist_ok=True)
    rel_path = os.path.relpath(args.target, os.path.dirname(args.yaml_out))
    # YOLO 표준: path 기준 상대. path 는 프로젝트 루트 기준 data/detector 로 명시.
    yaml_path = os.path.relpath(args.target, _PROJECT_ROOT).replace("\\", "/")
    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    yaml_text = (
        "# ============================================================\n"
        "# 1단계 검출기(YOLO) — 23종 다중 클래스 객체탐지 데이터셋 명세\n"
        "# scripts/build_detector_data.py 가 생성. 클래스 인덱스는\n"
        "# 영문 알파벳순(bibimbap=0 … tteokbokki=22), 분류기 23종 체크포인트와 동일.\n"
        "# ============================================================\n"
        f"path: {yaml_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(classes)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    with open(args.yaml_out, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    # ===== 리포트 =====
    def w(s, n, right=False):
        pad = max(0, n - sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(s)))
        s = str(s)
        return (" " * pad + s) if right else (s + " " * pad)

    print("\n" + "=" * 78)
    print("YOLO 23종 검출 데이터셋 — 클래스별 리포트")
    print("=" * 78)
    header = (w("idx", 4) + w("class", 18) + w("src", 7)
              + w("boxes", 7, True) + w("images", 8, True)
              + w("train", 7, True) + w("val", 6, True) + w("test", 6, True)
              + w("skip", 6, True) + w("noimg", 7, True))
    print(header)
    print("-" * len(header))
    tot = dict(boxes=0, images=0, train=0, val=0, test=0, skip=0, noimg=0)
    for i, c in enumerate(classes):
        tr, va, te = split_img[c]
        print(w(i, 4) + w(c, 18) + w(source_kind[c], 7)
              + w(box_counts[c], 7, True) + w(img_counts[c], 8, True)
              + w(tr, 7, True) + w(va, 6, True) + w(te, 6, True)
              + w(skipped_boxes[c], 6, True) + w(no_image[c], 7, True))
        tot["boxes"] += box_counts[c]; tot["images"] += img_counts[c]
        tot["train"] += tr; tot["val"] += va; tot["test"] += te
        tot["skip"] += skipped_boxes[c]; tot["noimg"] += no_image[c]
    print("-" * len(header))
    print(w("", 4) + w("합계", 18) + w("", 7)
          + w(tot["boxes"], 7, True) + w(tot["images"], 8, True)
          + w(tot["train"], 7, True) + w(tot["val"], 6, True) + w(tot["test"], 6, True)
          + w(tot["skip"], 6, True) + w(tot["noimg"], 7, True))
    print("=" * 78)
    print(f"이미지 합계: {tot['images']}  (train {tot['train']} / val {tot['val']} / test {tot['test']})")
    print(f"박스 합계: {tot['boxes']}  | skip된 박스: {tot['skip']}  | 라벨만 있고 이미지 없음: {tot['noimg']}")
    print(f"출력: {args.target}/images|labels/{{train,val,test}}/")
    print(f"명세: {args.yaml_out}")
    manual_total = sum(manual_counts.values())
    if manual_total:
        mc = {c: n for c, n in manual_counts.items() if n}
        print(f"수동(test 전용) 병합: {manual_total}장 {mc}  "
              f"(labels_manual/<class>/crop_area_manual.properties, train/val 제외)")


if __name__ == "__main__":
    main()
