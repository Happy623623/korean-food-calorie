"""classifier 추론 + 칼로리 매핑 파이프라인 (Phase 4-1).

학습한 분류기로 이미지 한 장을 받아 음식 종류 + 1인분 칼로리 + Top-K 예측을 반환한다.
나중에 Gradio 데모에서 `FoodClassifier(...).predict(image)` 형태로 호출한다.

사용 (프로젝트 루트에서):
    python -m src.classifier.inference --image data/classifier/test/gimbap/Img_001.jpg
    python -m src.classifier.inference                 # --image 생략 시 test/gimbap 첫 파일 자동 사용
    python -m src.classifier.inference --test-class bibimbap   # 해당 클래스 첫 파일 자동 사용

전처리는 학습/평가와 동일하게 dataset.build_transforms(image_size, train=False) 를
그대로 재사용한다(Resize -> Normalize -> ToTensorV2). 코드 중복/드리프트 방지.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import build_transforms          # val/test 변환 재사용 (단일 진실 공급원)
from .model import build_classifier
from ..utils import get_device, get_logger

logger = get_logger("korean-food.inference")

# 체크포인트마다 state_dict 키 이름이 다를 수 있어 우선순위대로 탐색한다.
# (evaluate.py 의 _STATE_DICT_KEYS 와 동일한 규약)
_STATE_DICT_KEYS = ("model", "model_state_dict", "state_dict")

# predict() 가 입력으로 받을 수 있는 타입
ImageLike = Union[str, os.PathLike, Image.Image, np.ndarray]


def _load_state_dict_into(model: torch.nn.Module, ckpt: Dict[str, Any]) -> None:
    """체크포인트 dict 에서 모델 가중치 키를 찾아 model 에 로드한다.

    train.py 는 'model' 키에 저장하지만, 다른 코드/버전 체크포인트도 받을 수 있도록
    흔한 키들을 순서대로 탐색한다. 하나도 없으면 키 목록과 함께 실패.
    """
    for k in _STATE_DICT_KEYS:
        if k in ckpt:
            model.load_state_dict(ckpt[k])
            return
    raise KeyError(f"체크포인트에 모델 가중치 키 없음. 키 목록: {list(ckpt.keys())}")


def _to_numpy_rgb(image: ImageLike) -> np.ndarray:
    """PIL.Image / numpy 배열 / 파일 경로를 HWC RGB uint8 numpy 배열로 변환한다.

    albumentations 변환은 numpy(HWC) 입력을 기대하므로 여기서 통일한다.
    """
    if isinstance(image, (str, os.PathLike)):
        return np.array(Image.open(image).convert("RGB"))

    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    if isinstance(image, np.ndarray):
        arr = image
        # 흑백(HW) -> 3채널로 확장
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        # RGBA -> RGB
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"지원하지 않는 numpy 이미지 형태: {arr.shape}")
        # float(0~1) 로 들어온 경우 uint8(0~255) 로 정규화
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).round()
            arr = arr.clip(0, 255).astype(np.uint8)
        return arr

    raise TypeError(
        f"지원하지 않는 이미지 입력 타입: {type(image)}. "
        "str/PathLike(경로), PIL.Image, numpy.ndarray 중 하나여야 합니다."
    )


class FoodClassifier:
    """학습된 분류기 + 칼로리 테이블을 묶은 추론 래퍼.

    이미지 한 장을 받아 음식 종류(top1) + 1인분 칼로리 + Top-K 예측을 반환한다.
    (model.py 의 nn.Module `FoodClassifier` 와 이름은 같지만 역할이 다른 상위 래퍼다.)
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/classifier_best.pt",
        food_table_path: str = "food_calories.json",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else get_device()

        # 1) 체크포인트 로드 (classes 등 비-텐서 메타데이터 포함 → weights_only=False)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        classes = ckpt.get("classes")
        if not classes:
            raise ValueError(
                f"체크포인트({checkpoint_path})에 classes 메타데이터가 없습니다. "
                "train.py 로 저장한 체크포인트인지 확인하세요."
            )
        self.classes: List[str] = list(classes)
        self.backbone: str = ckpt.get("backbone", "efficientnet_b0")
        self.image_size: int = int(ckpt.get("image_size", 224))
        num_classes = len(self.classes)

        # 2) 모델 빌드 (추론용: pretrained=False, dropout=0) + 가중치 로드
        self.model = build_classifier(
            backbone=self.backbone,
            num_classes=num_classes,
            pretrained=False,
            dropout=0.0,
        ).to(self.device)
        _load_state_dict_into(self.model, ckpt)
        self.model.eval()

        # 3) 칼로리 테이블 로드: english -> {korean, kcal_per_serving, ...}
        self.food_table: Dict[str, Dict[str, Any]] = self._load_food_table(food_table_path)

        # 4) 클래스 <-> 칼로리 테이블 정합성 검증 (안전장치)
        self._validate_classes()

        # 5) 전처리: 학습/평가의 val 변환을 그대로 재사용 (Resize→Normalize→ToTensorV2)
        self.transform = build_transforms(self.image_size, train=False)

        logger.info(
            "FoodClassifier 로드 완료 | device=%s | classes=%d | backbone=%s | image_size=%d",
            self.device, num_classes, self.backbone, self.image_size,
        )

    @staticmethod
    def _load_food_table(food_table_path: str) -> Dict[str, Dict[str, Any]]:
        """food_calories.json 을 읽어 english -> 칼로리 정보 dict 로 매핑한다."""
        with open(food_table_path, "r", encoding="utf-8") as f:
            raw: List[Dict[str, Any]] = json.load(f)
        table: Dict[str, Dict[str, Any]] = {}
        for item in raw:
            table[item["english"]] = {
                "korean": item.get("korean"),
                "kcal_per_serving": item.get("kcal_per_serving"),
                "serving_size_g": item.get("serving_size_g"),
                "kcal_per_100g": item.get("kcal_per_100g"),
                "source": item.get("source"),
            }
        return table

    def _validate_classes(self) -> None:
        """체크포인트 classes 와 food_calories.json english_names 일치 여부 검증.

        불일치(테이블에 없는 클래스)가 있어도 죽지 않고 경고만 남긴다.
        해당 클래스가 예측되면 predict() 가 칼로리 필드를 None 으로 채운다.
        """
        missing = [c for c in self.classes if c not in self.food_table]
        if missing:
            logger.warning(
                "칼로리 테이블에 없는 클래스 %d개: %s → 해당 클래스 예측 시 칼로리=None",
                len(missing), missing,
            )
        extra = [e for e in self.food_table if e not in self.classes]
        if extra:
            logger.warning(
                "칼로리 테이블에만 있고 모델 클래스엔 없는 항목 %d개(추론엔 무해): %s",
                len(extra), extra,
            )

    def _entry(self, english: str, confidence: float) -> Dict[str, Any]:
        """예측 클래스 1개에 대한 결과 entry(칼로리 정보 포함)를 만든다.

        칼로리 테이블에 없는 클래스면 칼로리 관련 필드는 None 으로 채운다.
        """
        info = self.food_table.get(english)
        return {
            "english": english,
            "korean": info["korean"] if info else None,
            "confidence": confidence,
            "kcal_per_serving": info["kcal_per_serving"] if info else None,
            "serving_size_g": info["serving_size_g"] if info else None,
            "kcal_per_100g": info["kcal_per_100g"] if info else None,
            "source": info["source"] if info else None,
        }

    @torch.no_grad()
    def predict(self, image: ImageLike, top_k: int = 3) -> Dict[str, Any]:
        """이미지 한 장을 추론해 top1 + top_k 예측(칼로리 포함)을 반환한다.

        Args:
            image: PIL.Image / numpy 배열(HWC) / 파일 경로 중 하나
            top_k: 반환할 상위 예측 개수(클래스 수로 자동 클램프)

        Returns:
            {"top1": {...}, "top_k": [{...}, ...]}  형태의 dict
        """
        arr = _to_numpy_rgb(image)
        tensor = self.transform(image=arr)["image"]          # (C, H, W) float tensor
        batch = tensor.unsqueeze(0).to(self.device)           # (1, C, H, W)

        logits = self.model(batch)                            # (1, num_classes)
        probs = F.softmax(logits, dim=1).squeeze(0)           # (num_classes,)

        k = max(1, min(top_k, len(self.classes)))
        top_probs, top_idx = torch.topk(probs, k)
        top_probs = top_probs.cpu().tolist()
        top_idx = top_idx.cpu().tolist()

        top_k_list = [
            self._entry(self.classes[idx], float(conf))
            for idx, conf in zip(top_idx, top_probs)
        ]

        logger.debug(
            "predict | top1=%s(%.4f) | k=%d",
            top_k_list[0]["english"], top_k_list[0]["confidence"], k,
        )

        return {"top1": top_k_list[0], "top_k": top_k_list}


# ======================================================================
# 테스트용 CLI
# ======================================================================
def _first_image_in(class_dir: str) -> Optional[str]:
    """주어진 폴더에서 첫 번째 이미지 파일 경로를 반환한다(없으면 None)."""
    if not os.path.isdir(class_dir):
        return None
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for fname in sorted(os.listdir(class_dir)):
        if fname.lower().endswith(exts):
            return os.path.join(class_dir, fname)
    return None


def _print_result(image_path: str, result: Dict[str, Any]) -> None:
    """predict 결과를 사람이 보기 좋게 출력한다."""
    top1 = result["top1"]
    print(f"\n입력 이미지 : {image_path}")
    print("=" * 56)
    kcal = top1["kcal_per_serving"]
    grams = top1["serving_size_g"]
    kcal_str = f"{kcal} kcal" if kcal is not None else "N/A (테이블에 없음)"
    grams_str = f"{grams} g" if grams is not None else "N/A"
    print(f"  음식      : {top1['korean']} ({top1['english']})")
    print(f"  신뢰도    : {top1['confidence'] * 100:.2f}%")
    print(f"  1인분     : {kcal_str}  /  {grams_str}")
    if top1["kcal_per_100g"] is not None:
        print(f"  100g당    : {top1['kcal_per_100g']} kcal  (출처: {top1['source']})")
    print("-" * 56)
    print(f"  Top-{len(result['top_k'])} 예측:")
    for rank, entry in enumerate(result["top_k"], start=1):
        kc = entry["kcal_per_serving"]
        kc_str = f"{kc} kcal" if kc is not None else "N/A"
        print(
            f"    {rank}. {entry['korean'] or '?'} ({entry['english']})"
            f"  {entry['confidence'] * 100:5.2f}%  |  {kc_str}"
        )
    print("=" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="음식 분류기 추론 + 칼로리 매핑 (이미지 한 장 테스트)"
    )
    parser.add_argument("--image", default=None,
                        help="추론할 이미지 경로. 생략 시 --test-class 폴더의 첫 파일을 자동 사용")
    parser.add_argument("--test-class", default="gimbap",
                        help="--image 생략 시 자동으로 첫 파일을 고를 test 클래스 폴더명")
    parser.add_argument("--test-root", default="data/classifier/test",
                        help="--image 생략 시 탐색할 test 데이터 루트")
    parser.add_argument("--checkpoint", default="checkpoints/classifier_best.pt",
                        help="분류기 체크포인트 경로")
    parser.add_argument("--food-table", default="food_calories.json",
                        help="칼로리 테이블 JSON 경로")
    parser.add_argument("--top-k", type=int, default=3, help="반환할 상위 예측 개수")
    args = parser.parse_args()

    image_path = args.image
    if image_path is None:
        class_dir = os.path.join(args.test_root, args.test_class)
        image_path = _first_image_in(class_dir)
        if image_path is None:
            parser.error(
                f"--image 미지정이고 '{class_dir}' 에서 이미지를 찾지 못했습니다. "
                "직접 --image 로 경로를 지정하세요."
            )
        logger.info("--image 미지정 → 자동 선택: %s", image_path)

    clf = FoodClassifier(
        checkpoint_path=args.checkpoint,
        food_table_path=args.food_table,
    )
    result = clf.predict(image_path, top_k=args.top_k)
    _print_result(image_path, result)


if __name__ == "__main__":
    main()
