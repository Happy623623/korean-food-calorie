"""추론 CLI (하이브리드 파이프라인 진입점).

이미지(또는 폴더)에 대해 YOLO 검출 -> EfficientNet 분류 -> 칼로리 합산을 수행한다.

사용법 (프로젝트 루트에서):
    # 결과를 콘솔(JSON)로 출력
    python infer.py --source path/to/food.jpg

    # 박스가 그려진 결과 이미지를 저장
    python infer.py --source path/to/food.jpg --save outputs/food_annotated.jpg

    # 폴더 전체를 처리 (이미지별 JSON 출력)
    python infer.py --source path/to/folder/
"""

import argparse
import json
import os

from src.pipeline import FoodCaloriePipeline
from src.utils import load_config

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _iter_sources(source: str):
    if os.path.isdir(source):
        for fname in sorted(os.listdir(source)):
            if fname.lower().endswith(IMG_EXTENSIONS):
                yield os.path.join(source, fname)
    else:
        yield source


def main() -> None:
    parser = argparse.ArgumentParser(description="음식 검출 + 분류 + 칼로리 추정 추론")
    parser.add_argument("--detector-config", default="configs/detector.yaml",
                        help="검출기 설정 파일 경로")
    parser.add_argument("--classifier-config", default="configs/classifier.yaml",
                        help="분류기 설정 파일 경로")
    parser.add_argument("--source", required=True, help="이미지 파일 또는 폴더 경로")
    parser.add_argument("--save", default=None,
                        help="박스가 그려진 결과 이미지 저장 경로(단일 이미지 입력 시)")
    args = parser.parse_args()

    pipeline = FoodCaloriePipeline(
        detector_config=load_config(args.detector_config),
        classifier_config=load_config(args.classifier_config),
    )

    for src in _iter_sources(args.source):
        if args.save and not os.path.isdir(args.source):
            result = pipeline.render(src, args.save)
            print(f"[infer] 주석 이미지 저장: {args.save}")
        else:
            result = pipeline.predict(src)
        print(f"\n=== {src} ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
