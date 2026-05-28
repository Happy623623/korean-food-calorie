"""분류기 학습용 데이터 준비 스크립트 (Phase 2-A).

AI Hub 한식 데이터셋(15종, 한글 폴더명)에서 이미지만 골라
food_calories.json 의 영문명으로 변환해 분류기 폴더 구조로 복사하고 8:1:1 로 분할한다.

    <source>/김밥/*.jpg  ─▶  <target>/train/gimbap/*.jpg
                              <target>/val/gimbap/*.jpg
                              <target>/test/gimbap/*.jpg

사용법 (프로젝트 루트에서):
    python -m src.classifier.prepare_data --source "C:\\Users\\yunwo\\Documents\\aihub_extracted"

동작:
    - food_calories.json 으로 한글명 -> 영문명 매핑을 만든다(단일 진실 공급원).
    - 소스의 각 한글 클래스 폴더에서 이미지(.jpg/.JPG/.jpeg)만 추려
      sklearn train_test_split 으로 8:1:1(기본) 분할 후 shutil.copy2 로 복사한다.
    - 비이미지 파일(crop_area.properties, org_url.csv 등)은 무시한다.
    - 끝나면 클래스별 train/val/test 개수를 표로 출력한다.

⚠️ 실제 이미지 데이터(AI Hub 압축 해제본)가 --source 경로에 준비되어 있어야 한다.
"""

import argparse
import os
import shutil
import sys
import unicodedata
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..utils import FoodTable

# 처리할 이미지 확장자(소문자 비교). .JPG/.JPEG 는 lower() 로 흡수된다.
IMG_EXTENSIONS = (".jpg", ".jpeg")

SPLITS = ("train", "val", "test")

# 프로젝트 루트 (src/classifier/prepare_data.py -> 상위 2단계)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_FOOD_TABLE = os.path.join(_PROJECT_ROOT, "food_calories.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Hub 한식 이미지를 분류기 폴더 구조로 복사/분할 (8:1:1)"
    )
    parser.add_argument(
        "--source", required=True,
        help="AI Hub 압축 해제 폴더 경로 (예: C:\\Users\\yunwo\\Documents\\aihub_extracted)",
    )
    parser.add_argument(
        "--target", default="data/classifier",
        help="저장할 경로 (기본값: data/classifier)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="학습 비율 (기본값: 0.8)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="검증 비율 (기본값: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="랜덤 시드 (기본값: 42)",
    )
    parser.add_argument(
        "--food-table", default=_DEFAULT_FOOD_TABLE,
        help="food_calories.json 경로 (기본값: 프로젝트 루트의 food_calories.json)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="타겟에 기존 데이터가 있어도 확인 없이 진행(기존 클래스 폴더 덮어씀)",
    )
    return parser.parse_args()


def list_images(class_dir: str) -> List[str]:
    """클래스 폴더 최상위에서 이미지 파일 경로만 정렬해 반환한다(비이미지 무시)."""
    images: List[str] = []
    for name in sorted(os.listdir(class_dir)):
        path = os.path.join(class_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMG_EXTENSIONS:
            images.append(path)
    return images


def split_files(
    files: List[str], train_ratio: float, val_ratio: float, seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """파일 목록을 train/val/test 로 분할한다(test = 1 - train - val).

    한 클래스 안에서의 분할이라 stratify 는 불필요하다.
    seed 를 random_state 로 넘겨 재현성을 보장한다.
    표본이 매우 적을 때(<=2장)는 train_test_split 의 0-분할 에러를 피하려 수동 처리한다.
    """
    n = len(files)
    if n == 0:
        return [], [], []
    if n == 1:
        return files, [], []
    if n == 2:
        # train 1, val 1, test 0 으로 최소 분할
        return files[:1], files[1:], []

    test_ratio = 1.0 - train_ratio - val_ratio
    train_files, rest = train_test_split(
        files, train_size=train_ratio, random_state=seed,
    )
    if len(rest) <= 1:
        return train_files, rest, []
    # 남은 val+test 비율 안에서 val 의 상대 비중
    rel_val = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.0
    val_files, test_files = train_test_split(
        rest, train_size=rel_val, random_state=seed,
    )
    return train_files, val_files, test_files


def target_has_data(target: str, english_names: List[str]) -> bool:
    """타겟에 이미 준비된(클래스 폴더 + 이미지) 데이터가 있는지 확인한다."""
    for split in SPLITS:
        for name in english_names:
            cls_dir = os.path.join(target, split, name)
            if os.path.isdir(cls_dir) and list_images(cls_dir):
                return True
    return False


def _display_width(text: str) -> int:
    """한글(전각) 2, ASCII 1 로 계산한 표시 폭(표 정렬용)."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    """표시 폭 기준으로 패딩한다(전각 문자 보정)."""
    pad = max(0, width - _display_width(text))
    return (" " * pad + text) if right else (text + " " * pad)


def print_summary(counts: Dict[str, Tuple[int, int, int]], name_w: int) -> None:
    """클래스별 train/val/test/합계 표를 출력한다."""
    num_w = 8
    header = (
        _pad("클래스", name_w)
        + _pad("Train", num_w, right=True)
        + _pad("Val", num_w, right=True)
        + _pad("Test", num_w, right=True)
        + _pad("합계", num_w, right=True)
    )
    print("\n" + header)
    print("-" * _display_width(header))

    t_tot = v_tot = s_tot = 0
    for name, (tr, va, te) in counts.items():
        total = tr + va + te
        t_tot += tr
        v_tot += va
        s_tot += te
        row = (
            _pad(name, name_w)
            + _pad(str(tr), num_w, right=True)
            + _pad(str(va), num_w, right=True)
            + _pad(str(te), num_w, right=True)
            + _pad(str(total), num_w, right=True)
        )
        print(row)

    print("-" * _display_width(header))
    grand = t_tot + v_tot + s_tot
    total_row = (
        _pad("합계", name_w)
        + _pad(str(t_tot), num_w, right=True)
        + _pad(str(v_tot), num_w, right=True)
        + _pad(str(s_tot), num_w, right=True)
        + _pad(str(grand), num_w, right=True)
    )
    print(total_row)


def main() -> None:
    args = parse_args()

    # --- 입력 검증 ---
    if args.train_ratio <= 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) >= 1.0:
        print(
            f"[오류] 비율이 올바르지 않습니다. 0 < train + val < 1 이어야 합니다 "
            f"(train={args.train_ratio}, val={args.val_ratio}, "
            f"test={1.0 - args.train_ratio - args.val_ratio:.2f}).",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isdir(args.source):
        print(f"[오류] 소스 폴더를 찾을 수 없습니다: {args.source}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.food_table):
        print(f"[오류] food_calories.json 을 찾을 수 없습니다: {args.food_table}", file=sys.stderr)
        sys.exit(1)

    # --- 한글명 -> 영문명 매핑 (단일 진실 공급원) ---
    table = FoodTable.from_json(args.food_table)
    kor_to_eng: Dict[str, str] = {info.korean: info.english for info in table.infos}
    english_names = table.english_names

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    print(
        f"[준비] source={args.source}\n"
        f"       target={args.target}\n"
        f"       분할 비율 train/val/test = "
        f"{args.train_ratio:.2f}/{args.val_ratio:.2f}/{test_ratio:.2f} | seed={args.seed}"
    )

    # --- 안전장치: 타겟에 기존 데이터가 있으면 확인 ---
    if target_has_data(args.target, english_names) and not args.force:
        ans = input(
            f"\n[확인] 타겟에 이미 준비된 데이터가 있습니다: {args.target}\n"
            f"       기존 클래스 폴더를 덮어쓰고 진행할까요? [y/N] "
        ).strip().lower()
        if ans not in ("y", "yes"):
            print("[취소] 사용자 요청으로 중단합니다.")
            sys.exit(0)

    # --- 클래스별 복사 ---
    counts: Dict[str, Tuple[int, int, int]] = {}
    missing: List[str] = []
    name_w = max([_display_width("클래스")] + [len(n) for n in english_names]) + 2

    for korean in tqdm(table.korean_names, desc="클래스 처리", unit="class"):
        english = kor_to_eng[korean]
        src_dir = os.path.join(args.source, korean)

        if not os.path.isdir(src_dir):
            missing.append(f"{korean}({english})")
            counts[english] = (0, 0, 0)
            continue

        images = list_images(src_dir)
        if not images:
            missing.append(f"{korean}({english}) - 이미지 0장")
            counts[english] = (0, 0, 0)
            continue

        tr_files, va_files, te_files = split_files(
            images, args.train_ratio, args.val_ratio, args.seed,
        )

        for split, files in zip(SPLITS, (tr_files, va_files, te_files)):
            dst_dir = os.path.join(args.target, split, english)
            # 재실행 시 stale 파일이 섞이지 않도록 해당 클래스 폴더를 새로 만든다
            if os.path.isdir(dst_dir):
                shutil.rmtree(dst_dir)
            os.makedirs(dst_dir, exist_ok=True)
            for src_path in tqdm(files, desc=f"  {english}/{split}", unit="img", leave=False):
                shutil.copy2(src_path, os.path.join(dst_dir, os.path.basename(src_path)))

        counts[english] = (len(tr_files), len(va_files), len(te_files))

    # --- 결과 보고 ---
    print_summary(counts, name_w)

    if missing:
        print(f"\n[경고] 소스에 없거나 비어 있어 건너뛴 클래스 {len(missing)}개:")
        for item in missing:
            print(f"   - {item}")

    print(f"\n[완료] 분류기 데이터 준비 완료 -> {args.target}/{{train,val,test}}/<english_name>/")


if __name__ == "__main__":
    main()
