"""분류기 학습용 데이터 준비 스크립트 (Phase 2-A → 23종 멀티소스).

여러 출처(AI Hub kfood 150종 / UEC FOOD-256)에서 모은 한글 폴더 이미지를
food_calories.json 의 영문명·인덱스 순서로 변환해 분류기 폴더 구조로 만들고
8:1:1 로 층화 분할한다.

    <source>/김밥/*.jpg        ─▶  <target>/train/gimbap/*.jpg   (kfood: 원본)
    <source>/돈까스/*.jpg + bb_info.txt ─▶ <target>/train/donkkaseu/*.jpg (UEC: 박스 크롭)

출처별 처리:
    - UEC 클래스(폴더에 bb_info.txt 존재: 돈까스·밥·소시지·스파게티·햄버거):
        bb_info.txt 의 박스로 크롭(박스 여러 개면 가장 큰 것, 없으면 원본).
    - 그 외 kfood 클래스: 원본 전체 이미지(크롭 안 함).
    - 공통: 짧은 변을 256px 로 리사이즈(비율 유지, 이미 작으면 그대로) 후 jpg 저장.

사용법 (프로젝트 루트에서):
    python -m src.classifier.prepare_data --source "C:\\Users\\yunwo\\Documents\\aihub_extracted"

동작 요약:
    1) 검증: 소스 폴더 목록 ↔ 23개 매핑이 1:1 인지, food_calories.json 에 23개
       english 키가 다 있는지 확인. 불일치 시 이름 출력 후 중단.
    2) data/classifier 가 있으면 통째로 지우고 23종으로 새로 생성.
    3) 클래스별 8:1:1 층화 분할(seed 고정) → 이미지 처리 후 저장.
    4) train split 기준 역빈도 클래스 가중치 → configs/class_weights.json 저장.
    5) 클래스별 total/train/val/test + 가중치 표 출력.

⚠️ 실제 이미지 데이터가 --source 경로에 준비되어 있어야 한다.
"""

import argparse
import json
import os
import shutil
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..utils import FoodTable

# 처리할 이미지 확장자(소문자 비교, 대소문자 무시). .JPG/.PNG 등은 lower() 로 흡수.
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

# 비이미지/메타 파일 (글롭에서 자동 제외되지만 명시적으로 기록)
META_FILENAMES = ("bb_info.txt", "crop_area.properties", "org_url.csv")

SPLITS = ("train", "val", "test")

RESIZE_SHORT_SIDE = 256  # 짧은 변 목표 길이(px)

# 프로젝트 루트 (src/classifier/prepare_data.py -> 상위 2단계)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_FOOD_TABLE = os.path.join(_PROJECT_ROOT, "food_calories.json")
_DEFAULT_WEIGHTS = os.path.join(_PROJECT_ROOT, "configs", "class_weights.json")

# ---------------------------------------------------------------------------
# 신규 8개 클래스: 실제 소스 폴더명(한글) → 영문명.
# food_calories.json 의 korean 표기와 폴더명이 다른 경우가 있어(달걀프라이↔계란후라이,
# 소세지↔소시지) 여기서 '실제 폴더명' 기준으로 못박는다. 기존 15개는 JSON 의
# korean 이 폴더명과 일치하므로 JSON 에서 그대로 가져온다.
# ---------------------------------------------------------------------------
NEW_FOLDER_TO_ENGLISH: Dict[str, str] = {
    "짬뽕": "jjamppong",
    "계란후라이": "fried-egg",
    "피자": "pizza",
    "돈까스": "donkkaseu",
    "밥": "rice",
    "소시지": "sausage",
    "스파게티": "spaghetti",
    "햄버거": "hamburger",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="멀티소스(kfood/UEC) 한식 이미지를 23종 분류기 폴더로 처리/분할 (8:1:1)"
    )
    parser.add_argument(
        "--source", required=True,
        help="압축 해제 폴더 경로 (예: C:\\Users\\yunwo\\Documents\\aihub_extracted)",
    )
    parser.add_argument(
        "--target", default="data/classifier",
        help="저장할 경로 (기본값: data/classifier)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="학습 비율 (기본 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="검증 비율 (기본 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (기본 42)")
    parser.add_argument(
        "--food-table", default=_DEFAULT_FOOD_TABLE,
        help="food_calories.json 경로 (기본: 프로젝트 루트)",
    )
    parser.add_argument(
        "--weights-out", default=_DEFAULT_WEIGHTS,
        help="클래스 가중치 저장 경로 (기본: configs/class_weights.json)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 유니코드/표 출력 헬퍼
# ---------------------------------------------------------------------------
def nfc(text: str) -> str:
    """한글 폴더명 비교용 정규화(NFC). 일부 FS 는 NFD 로 저장해 비교가 어긋날 수 있다."""
    return unicodedata.normalize("NFC", text)


def _display_width(text: str) -> int:
    """한글(전각) 2, ASCII 1 로 계산한 표시 폭(표 정렬용)."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    pad = max(0, width - _display_width(text))
    return (" " * pad + text) if right else (text + " " * pad)


# ---------------------------------------------------------------------------
# 매핑 구성 & 검증
# ---------------------------------------------------------------------------
def build_folder_map(table: FoodTable) -> Dict[str, str]:
    """소스 '폴더명(한글, NFC) → 영문명' 매핑 23개를 만든다.

    기존 15개는 food_calories.json 의 korean(=폴더명)에서, 신규 8개는
    NEW_FOLDER_TO_ENGLISH(실제 폴더명)에서 가져온다.
    """
    new_english = set(NEW_FOLDER_TO_ENGLISH.values())
    folder_to_eng: Dict[str, str] = {
        nfc(info.korean): info.english
        for info in table.infos
        if info.english not in new_english  # 신규 english 는 JSON korean(달걀프라이 등) 대신 폴더명 사용
    }
    for kor, eng in NEW_FOLDER_TO_ENGLISH.items():
        folder_to_eng[nfc(kor)] = eng
    return folder_to_eng


def validate(source: str, table: FoodTable, folder_to_eng: Dict[str, str]) -> None:
    """소스 폴더 ↔ 매핑 1:1, food_calories.json 의 23개 english 키 존재를 검증.

    불일치 시 사유를 출력하고 즉시 종료한다.
    """
    expected_english = list(NEW_FOLDER_TO_ENGLISH.values()) + [
        info.english for info in table.infos
        if info.english not in set(NEW_FOLDER_TO_ENGLISH.values())
    ]

    # (a) food_calories.json 에 23개 english 키가 다 있는지
    json_english = set(table.english_names)
    mapped_english = set(folder_to_eng.values())
    print("[검증] food_calories.json 클래스 수:", len(table.infos))
    if len(table.infos) != 23:
        print(f"[중단] food_calories.json 항목이 23개가 아닙니다: {len(table.infos)}개", file=sys.stderr)
        sys.exit(1)
    miss_in_json = mapped_english - json_english
    if miss_in_json:
        print(f"[중단] 매핑 english 중 food_calories.json 에 없는 키: {sorted(miss_in_json)}", file=sys.stderr)
        sys.exit(1)
    print("[검증] 23개 english 키 모두 food_calories.json 에 존재 ✓")

    # (b) 소스 폴더 ↔ 매핑(23) 정확히 1:1
    source_dirs = {
        nfc(d) for d in os.listdir(source)
        if os.path.isdir(os.path.join(source, d))
    }
    expected_dirs = set(folder_to_eng.keys())
    missing = expected_dirs - source_dirs   # 매핑엔 있는데 소스에 없음
    extra = source_dirs - expected_dirs     # 소스엔 있는데 매핑에 없음(초과)
    print(f"[검증] 소스 폴더 수: {len(source_dirs)} / 매핑 클래스 수: {len(expected_dirs)}")
    if missing or extra:
        if missing:
            print(f"[중단] 소스에서 누락된 폴더 {len(missing)}개: {sorted(missing)}", file=sys.stderr)
        if extra:
            print(f"[중단] 매핑에 없는 초과 폴더 {len(extra)}개: {sorted(extra)}", file=sys.stderr)
        sys.exit(1)
    print("[검증] 소스 폴더 ↔ 23개 매핑 정확히 1:1 ✓\n")


# ---------------------------------------------------------------------------
# 이미지 글롭 / bb_info 파싱 / 처리
# ---------------------------------------------------------------------------
def list_images(class_dir: str) -> List[str]:
    """클래스 폴더 최상위에서 이미지 파일 경로만 정렬해 반환(메타/비이미지 제외)."""
    images: List[str] = []
    for name in sorted(os.listdir(class_dir)):
        path = os.path.join(class_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMG_EXTENSIONS:
            images.append(path)
    return images


def is_uec_class(class_dir: str) -> bool:
    """UEC 클래스 여부 = 폴더에 bb_info.txt 존재."""
    return os.path.isfile(os.path.join(class_dir, "bb_info.txt"))


def parse_bb_info(class_dir: str) -> Dict[str, Tuple[int, int, int, int]]:
    """bb_info.txt 를 파싱해 {이미지ID(str): 가장 큰 박스(x1,y1,x2,y2)} 를 만든다.

    형식: 헤더 'img x1 y1 x2 y2' 이후 각 줄 '<id> x1 y1 x2 y2'.
    같은 id 가 여러 줄(박스 여러 개)이면 넓이가 가장 큰 박스를 채택한다.
    """
    path = os.path.join(class_dir, "bb_info.txt")
    boxes: Dict[str, Tuple[int, int, int, int]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f):
            parts = line.split()
            if len(parts) != 5:
                continue
            if ln == 0 and parts[0].lower() == "img":  # 헤더
                continue
            try:
                img_id = str(int(parts[0]))
                x1, y1, x2, y2 = (int(v) for v in parts[1:])
            except ValueError:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            prev = boxes.get(img_id)
            if prev is None or area > (prev[2] - prev[0]) * (prev[3] - prev[1]):
                boxes[img_id] = (x1, y1, x2, y2)
    return boxes


def process_image(
    src_path: str,
    box: Optional[Tuple[int, int, int, int]],
    dst_path: str,
) -> bool:
    """이미지를 (선택)크롭 → 짧은 변 256 리사이즈 → jpg 저장. 성공 시 True."""
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            if box is not None:
                w, h = im.size
                x1 = max(0, min(box[0], w - 1))
                y1 = max(0, min(box[1], h - 1))
                x2 = max(x1 + 1, min(box[2], w))
                y2 = max(y1 + 1, min(box[3], h))
                im = im.crop((x1, y1, x2, y2))
            w, h = im.size
            short = min(w, h)
            if short > RESIZE_SHORT_SIDE:  # 이미 작으면 그대로
                scale = RESIZE_SHORT_SIDE / short
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            im.save(dst_path, "JPEG", quality=92)
        return True
    except Exception as e:  # 손상 이미지 등은 건너뛴다
        print(f"   [skip] 처리 실패: {src_path} ({e})", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# 분할
# ---------------------------------------------------------------------------
def split_files(
    files: List[str], train_ratio: float, val_ratio: float, seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """파일 목록을 train/val/test 로 분할(test = 1 - train - val). 표본 적을 때 수동 처리."""
    n = len(files)
    if n == 0:
        return [], [], []
    if n == 1:
        return files, [], []
    if n == 2:
        return files[:1], files[1:], []

    test_ratio = 1.0 - train_ratio - val_ratio
    train_files, rest = train_test_split(files, train_size=train_ratio, random_state=seed)
    if len(rest) <= 1:
        return train_files, rest, []
    rel_val = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.0
    val_files, test_files = train_test_split(rest, train_size=rel_val, random_state=seed)
    return train_files, val_files, test_files


# ---------------------------------------------------------------------------
# 가중치 & 리포트
# ---------------------------------------------------------------------------
def compute_class_weights(train_counts: Dict[str, int]) -> Dict[str, float]:
    """train split 기준 역빈도 가중치(sklearn 'balanced' 방식).

        w_c = n_samples / (n_classes * count_c)
    평균이 대략 1 이 되도록 정규화된다. count 0 인 클래스는 0.0.
    """
    n_classes = len(train_counts)
    n_samples = sum(train_counts.values())
    weights: Dict[str, float] = {}
    for eng, cnt in train_counts.items():
        weights[eng] = (n_samples / (n_classes * cnt)) if cnt > 0 else 0.0
    return weights


def print_summary(
    order: List[str],
    counts: Dict[str, Tuple[int, int, int]],
    weights: Dict[str, float],
    name_w: int,
) -> None:
    """클래스별 total/train/val/test + 가중치 표를 출력."""
    num_w = 8
    wt_w = 10
    header = (
        _pad("클래스", name_w)
        + _pad("Total", num_w, right=True)
        + _pad("Train", num_w, right=True)
        + _pad("Val", num_w, right=True)
        + _pad("Test", num_w, right=True)
        + _pad("Weight", wt_w, right=True)
    )
    print("\n" + header)
    print("-" * _display_width(header))

    t_tot = v_tot = s_tot = g_tot = 0
    for name in order:
        tr, va, te = counts.get(name, (0, 0, 0))
        total = tr + va + te
        t_tot += tr; v_tot += va; s_tot += te; g_tot += total
        print(
            _pad(name, name_w)
            + _pad(str(total), num_w, right=True)
            + _pad(str(tr), num_w, right=True)
            + _pad(str(va), num_w, right=True)
            + _pad(str(te), num_w, right=True)
            + _pad(f"{weights.get(name, 0.0):.4f}", wt_w, right=True)
        )

    print("-" * _display_width(header))
    print(
        _pad("합계", name_w)
        + _pad(str(g_tot), num_w, right=True)
        + _pad(str(t_tot), num_w, right=True)
        + _pad(str(v_tot), num_w, right=True)
        + _pad(str(s_tot), num_w, right=True)
        + _pad("-", wt_w, right=True)
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- 입력 검증 ---
    if args.train_ratio <= 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) >= 1.0:
        print(
            f"[오류] 비율이 올바르지 않습니다. 0 < train + val < 1 이어야 합니다 "
            f"(train={args.train_ratio}, val={args.val_ratio}).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isdir(args.source):
        print(f"[오류] 소스 폴더를 찾을 수 없습니다: {args.source}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.food_table):
        print(f"[오류] food_calories.json 을 찾을 수 없습니다: {args.food_table}", file=sys.stderr)
        sys.exit(1)

    table = FoodTable.from_json(args.food_table)
    english_order = table.english_names  # 클래스 인덱스 순서 (food_calories.json 등장 순서)
    folder_to_eng = build_folder_map(table)

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    print(
        f"[준비] source={args.source}\n"
        f"       target={args.target}\n"
        f"       분할 train/val/test = "
        f"{args.train_ratio:.2f}/{args.val_ratio:.2f}/{test_ratio:.2f} | seed={args.seed}\n"
    )

    # ===== 1) 검증 (먼저) =====
    validate(args.source, table, folder_to_eng)

    # english → 소스 폴더명(NFC)
    eng_to_folder = {v: k for k, v in folder_to_eng.items()}

    # ===== 2) data/classifier 통째로 새로 생성 =====
    if os.path.isdir(args.target):
        print(f"[초기화] 기존 타겟 삭제: {args.target}")
        shutil.rmtree(args.target)
    for split in SPLITS:
        os.makedirs(os.path.join(args.target, split), exist_ok=True)

    # 소스 폴더명을 NFC->실제 dirname 으로 되돌리기 위한 맵(파일 접근용)
    actual_dirname = {nfc(d): d for d in os.listdir(args.source)
                      if os.path.isdir(os.path.join(args.source, d))}

    # ===== 3) 클래스별 처리 & 분할 =====
    counts: Dict[str, Tuple[int, int, int]] = {}
    train_counts: Dict[str, int] = {}
    uec_classes: List[str] = []
    name_w = max([_display_width("클래스")] + [len(n) for n in english_order]) + 2

    for english in tqdm(english_order, desc="클래스 처리", unit="class"):
        folder_nfc = eng_to_folder[english]
        src_dir = os.path.join(args.source, actual_dirname[folder_nfc])

        images = list_images(src_dir)
        uec = is_uec_class(src_dir)
        boxes = parse_bb_info(src_dir) if uec else {}
        if uec:
            uec_classes.append(english)

        tr_files, va_files, te_files = split_files(
            images, args.train_ratio, args.val_ratio, args.seed,
        )

        saved = {"train": 0, "val": 0, "test": 0}
        for split, files in zip(SPLITS, (tr_files, va_files, te_files)):
            dst_dir = os.path.join(args.target, split, english)
            os.makedirs(dst_dir, exist_ok=True)
            for src_path in tqdm(files, desc=f"  {english}/{split}", unit="img", leave=False):
                stem = os.path.splitext(os.path.basename(src_path))[0]
                box = boxes.get(str(stem)) if uec else None  # bb 에 없으면 None → 원본
                dst_path = os.path.join(dst_dir, stem + ".jpg")
                if process_image(src_path, box, dst_path):
                    saved[split] += 1

        counts[english] = (saved["train"], saved["val"], saved["test"])
        train_counts[english] = saved["train"]

    # ===== 4) 클래스 가중치 → configs/class_weights.json =====
    weights = compute_class_weights(train_counts)
    os.makedirs(os.path.dirname(args.weights_out), exist_ok=True)
    with open(args.weights_out, "w", encoding="utf-8") as f:
        json.dump(
            {eng: round(weights[eng], 6) for eng in english_order},
            f, ensure_ascii=False, indent=2,
        )

    # ===== 5) 리포트 =====
    print_summary(english_order, counts, weights, name_w)
    print(f"\n[출처] UEC(bb_info 크롭) 클래스 {len(uec_classes)}개: {uec_classes}")
    print(f"       그 외 {len(english_order) - len(uec_classes)}개: kfood 원본(크롭 없음)")
    print(f"[가중치] train 역빈도 가중치 저장 → {args.weights_out}")
    print(f"[완료] 분류기 데이터 준비 완료 -> {args.target}/{{train,val,test}}/<english>/")


if __name__ == "__main__":
    main()
