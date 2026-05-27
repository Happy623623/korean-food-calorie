"""detector 학습: Ultralytics YOLO11 '음식 영역' 검출기 (하이브리드 1단계).

사용법 (프로젝트 루트에서):
    python -m src.detector.train_yolo --config configs/detector.yaml

동작:
    1) detector.yaml 의 classes/데이터 경로로부터 ultralytics 용 dataset.yaml 을 자동 생성
       (ultralytics 의 상대경로 처리 함정을 피하기 위해 절대경로로 기록)
    2) base_weights(예: yolo11s.pt, COCO 사전학습)에서 fine-tuning
    3) 학습 완료 후 best.pt 를 config 의 model.weights 경로로 복사

⚠️ 데이터(이미지 + YOLO 라벨)가 준비되어 있어야 한다.
   data/detector/images/<split>/*.jpg  와  data/detector/labels/<split>/*.txt 가
   1:1 대응해야 하며, 라벨 한 줄은 "<class_id> <cx> <cy> <w> <h>" (0~1 정규화) 형식이다.
   하이브리드 기본 설정에서는 클래스가 'food' 하나뿐이므로 class_id 는 모두 0 이다.
"""

import argparse
import os
import shutil
from typing import Any, Dict

import yaml

from ..utils import load_config, set_seed


def write_dataset_yaml(config: Dict[str, Any]) -> str:
    """config 로부터 ultralytics 데이터셋 명세(dataset.yaml)를 생성하고 경로를 반환한다.

    classes 는 detector.yaml 한 곳에서만 관리하고 여기서 idx->name 매핑으로 펼친다.
    path 는 절대경로로 기록해 ultralytics 의 datasets_dir 기준 상대경로 해석을 우회한다.
    """
    data_cfg = config["data"]
    root_abs = os.path.abspath(data_cfg["root"])
    classes = config["classes"]

    spec = {
        "path": root_abs,
        "train": data_cfg["train"],
        "val": data_cfg["val"],
        "test": data_cfg["test"],
        "names": {idx: name for idx, name in enumerate(classes)},
    }

    out_path = data_cfg["dataset_yaml"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, allow_unicode=True, sort_keys=False)
    return out_path


def resolve_device(requested: Any) -> Any:
    """요청된 학습 device 를 확정한다.

    config/CLI 에서 GPU(0, "0", "cuda", "cuda:0", "" 자동)를 요청했더라도
    CUDA 가 사용 불가능하면 자동으로 "cpu" 로 fallback 한다.
    명시적으로 "cpu" 를 요청한 경우는 그대로 둔다.
    """
    import torch  # 지연 임포트

    if str(requested).strip().lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        print(f"[train_yolo] CUDA 미가용 -> 요청 device={requested!r} 무시하고 cpu 로 fallback")
        return "cpu"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11 음식 영역 검출기 학습 (하이브리드 1단계)")
    parser.add_argument("--config", default="configs/detector.yaml", help="설정 파일 경로")
    parser.add_argument("--epochs", type=int, default=None, help="config 값 덮어쓰기")
    parser.add_argument("--batch", type=int, default=None, help="config 값 덮어쓰기")
    parser.add_argument("--device", default=None, help='"", "0", "cpu" 등')
    parser.add_argument("--base-weights", default=None, help="사전학습 가중치 덮어쓰기")
    args = parser.parse_args()

    from ultralytics import YOLO  # 지연 임포트

    config = load_config(args.config)
    set_seed(config.get("project", {}).get("seed", 42))

    m = config["model"]
    dataset_yaml = write_dataset_yaml(config)
    print(f"[train_yolo] 데이터셋 명세 생성: {dataset_yaml}")

    base_weights = args.base_weights or m["base_weights"]
    model = YOLO(base_weights)

    # device: CLI > config(기본 0=첫 GPU). CUDA 미가용 시 cpu 로 자동 fallback.
    requested_device = args.device if args.device is not None else m.get("device", 0)
    device = resolve_device(requested_device)
    print(f"[train_yolo] device={device}")

    results = model.train(
        data=dataset_yaml,
        imgsz=int(m.get("imgsz", 640)),
        epochs=int(args.epochs if args.epochs is not None else m.get("epochs", 100)),
        batch=int(args.batch if args.batch is not None else m.get("batch", 16)),
        device=device,
        workers=int(m.get("workers", 8)),
        patience=int(m.get("patience", 20)),
        project=m.get("project", "runs"),
        name=m.get("name", "yolo11_food_detector"),
        seed=config.get("project", {}).get("seed", 42),
    )

    # 학습 결과의 best.pt 를 추론용 가중치 경로로 복사
    best = os.path.join(str(results.save_dir), "weights", "best.pt")
    dest = m["weights"]
    if os.path.isfile(best):
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        shutil.copy2(best, dest)
        print(f"[train_yolo] 완료. best.pt -> {dest}")
    else:
        print(f"[train_yolo] 주의: best.pt 를 찾지 못했습니다 ({best}). save_dir 를 확인하세요.")


if __name__ == "__main__":
    main()
