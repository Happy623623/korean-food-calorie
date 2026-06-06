"""detector 학습: Ultralytics YOLO11n '음식 23종' 검출기 (하이브리드 1단계).

Colab 학습 레시피와 동일:
    yolo11n.pt + COCO 사전학습 → configs/detector_data.yaml(nc=23) 로 fine-tuning,
    epochs=100, patience=20, imgsz=640, batch=32, seed=42, 나머지 ultralytics 기본값.

사용법 (프로젝트 루트에서):
    python -m src.detector.train_yolo --config configs/detector.yaml

동작:
    1) 데이터셋 명세 결정: config.data.dataset_spec(=configs/detector_data.yaml) 가 있으면
       그대로 사용(Colab 레시피와 동일). 없으면 detector.yaml 의 classes 로 자동 생성.
    2) base_weights(yolo11n.pt, COCO 사전학습)에서 fine-tuning.
    3) 학습 완료 후 best.pt 를 config 의 model.weights 경로로 복사.

⚠️ 데이터(이미지 + YOLO 라벨)가 준비되어 있어야 한다.
   data/detector/images/<split>/*.jpg  와  data/detector/labels/<split>/*.txt 가
   1:1 대응해야 하며, 라벨 한 줄은 "<class_id> <cx> <cy> <w> <h>" (0~1 정규화) 형식이다.
   class_id 는 0~22(영문 알파벳순 23종).
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


def resolve_dataset_spec(config: Dict[str, Any]) -> str:
    """학습에 쓸 ultralytics 데이터셋 명세 경로를 결정한다.

    Colab 레시피와 동일하게 configs/detector_data.yaml(nc=23) 가 있으면 그대로 쓰고,
    없으면 detector.yaml 의 classes 로 dataset.yaml 을 자동 생성한다.
    """
    spec = config.get("data", {}).get("dataset_spec")
    if spec and os.path.isfile(spec):
        print(f"[train_yolo] 데이터셋 명세 사용(레시피 일치): {spec}")
        return spec
    generated = write_dataset_yaml(config)
    print(f"[train_yolo] 데이터셋 명세 자동 생성: {generated}")
    return generated


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
    dataset_yaml = resolve_dataset_spec(config)

    base_weights = args.base_weights or m["base_weights"]
    model = YOLO(base_weights)

    # device: CLI > config(기본 0=첫 GPU). CUDA 미가용 시 cpu 로 자동 fallback.
    requested_device = args.device if args.device is not None else m.get("device", 0)
    device = resolve_device(requested_device)
    print(f"[train_yolo] device={device}")

    # Colab 레시피와 동일한 핵심 하이퍼파라미터만 명시하고 나머지는 ultralytics 기본값.
    results = model.train(
        data=dataset_yaml,
        imgsz=int(m.get("imgsz", 640)),
        epochs=int(args.epochs if args.epochs is not None else m.get("epochs", 100)),
        batch=int(args.batch if args.batch is not None else m.get("batch", 32)),
        patience=int(m.get("patience", 20)),
        device=device,
        project=m.get("project", "runs"),
        name=m.get("name", "yolo11n_food_23cls"),
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
