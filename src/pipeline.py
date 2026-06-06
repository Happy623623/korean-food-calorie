"""하이브리드 추론 파이프라인 (검출 박스 수에 따라 단일/다중 경로 분기).

    이미지 ──▶ [1단계: YOLO11 검출(conf=0.35, CPU)] ──▶ 음식 영역 bbox 들
                                        │
            ┌───────────────────────────┴────────────────────────────┐
            ▼ 박스 0~1개                                              ▼ 박스 ≥2개
   [단일 경로] 전체 이미지 → EfficientNet 분류            [다중 경로] 박스별 크롭(각 변 10%
   (= 기존 검증된 동작 그대로 보존)                       패딩·경계 clip) → EfficientNet 분류
            │                                                        │
            ▼                                                        ▼
   음식 1종 + 1인분 kcal                          항목별 음식/kcal + 총합(중복 박스 제외)

설계 원칙:
  - 최종 음식 "종류"는 항상 2단계 분류기(EfficientNet)가 결정한다.
    YOLO 가 본 종류(yolo_class)는 참고 표시용으로만 함께 반환한다.
  - 박스 1개당 "1인분 평균 kcal"로 합산한다(portion 회귀 없음).
  - 다중 경로에서 클래스 무관 IoU>0.6 으로 겹치는 박스는 conf 높은 것만 남겨
    같은 음식을 두 번 세어 칼로리가 이중 합산되는 것을 막는다.

torch / ultralytics / opencv 의존성은 파이프라인을 실제로 생성할 때만 로드된다(지연 임포트).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .utils import FoodTable, get_device

# 다중 경로 진입 기준(검출 박스 수)과 중복 제거 IoU 임계값.
MULTI_MIN_BOXES = 2
DEDUP_IOU_THRESHOLD = 0.6
# 다중 경로에서 각 박스를 분류 전 좌/우/상/하로 넓히는 비율(상자가 음식을 살짝 잘라도 보강).
CROP_PADDING_RATIO = 0.10


class FoodCaloriePipeline:
    """YOLO 검출 + EfficientNet 분류 + 칼로리 합산을 묶은 엔드투엔드 추론기."""

    def __init__(
        self,
        detector_config: Dict[str, Any],
        classifier_config: Dict[str, Any],
    ) -> None:
        import torch  # 지연 임포트

        from .classifier.dataset import build_transforms
        from .classifier.model import build_classifier
        from .detector.inference_yolo import FoodDetector

        self._torch = torch
        self.device = get_device()

        # --- 1단계: 검출기 (박스만, 종류 분류 안 함) ---
        self.detector = FoodDetector(detector_config)

        # --- 칼로리/클래스 단일 진실 공급원 ---
        food_table_path = classifier_config["food_table"]
        self.table = FoodTable.from_json(food_table_path)
        # source / kcal_per_100g 등 부가 필드는 raw json 에서 직접 읽는다(FoodInfo 에는 없음).
        self.eng_to_full = self._load_full_table(food_table_path)

        # --- 2단계: 분류기 ---
        cm = classifier_config["model"]
        weights = classifier_config["train"]["weights"]
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"분류기 가중치를 찾을 수 없습니다: {weights}\n"
                f"먼저 `python -m src.classifier.train` 로 학습하세요."
            )

        checkpoint = torch.load(weights, map_location=self.device, weights_only=False)
        # 체크포인트에 저장된 메타데이터로 클래스 정렬/입력 크기를 복원.
        # classes 는 학습 시 ImageFolder 알파벳 순일 수 있으므로 JSON 순서에 의존하지 않고
        # 반드시 이 리스트(인덱스→영문명)를 사용한다.
        self.classes: List[str] = checkpoint.get("classes", self.table.english_names)
        backbone = checkpoint.get("backbone", cm.get("backbone", "efficientnet_b0"))
        self.image_size = int(checkpoint.get("image_size", cm.get("image_size", 224)))
        num_classes = int(checkpoint.get("num_classes", len(self.classes)))

        # 영문명 → FoodInfo 조회 테이블(칼로리 매핑은 인덱스가 아닌 영문 키로 한다).
        self.eng_to_info = {info.english: info for info in self.table.infos}

        # 정합성 검증: 모델 클래스가 칼로리 테이블에 모두 존재하는지(순서는 무관).
        missing = [c for c in self.classes if c not in self.eng_to_info]
        if missing:
            raise ValueError(
                "분류기 체크포인트의 클래스가 food_calories.json 에 없습니다: "
                f"{missing}\n동일한 food_calories.json 으로 분류기를 학습/저장하세요."
            )

        self.model = build_classifier(
            backbone=backbone, num_classes=num_classes, pretrained=False, dropout=0.0,
        )
        # 체크포인트마다 가중치 키가 다를 수 있어 우선순위대로 탐색(state_dict/model/...).
        state_dict = None
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(checkpoint, dict) and key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint  # 순수 state_dict 로 저장된 경우
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self._transform = build_transforms(self.image_size, train=False)

    # ------------------------------------------------------------------
    # 생성 헬퍼
    # ------------------------------------------------------------------
    @classmethod
    def from_paths(
        cls,
        detector_weights: str = "checkpoints/yolo11n_23cls.pt",
        classifier_weights: str = "checkpoints/best_efficientnet_b0_23.pt",
        food_table: str = "food_calories.json",
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        infer_device: str = "cpu",
    ) -> "FoodCaloriePipeline":
        """가중치 경로만으로 파이프라인을 만든다(app.py / 스모크용 간편 생성자)."""
        detector_config = {
            "model": {
                "weights": detector_weights,
                "conf": conf,
                "iou": iou,
                "imgsz": imgsz,
                "infer_device": infer_device,
            }
        }
        classifier_config = {
            "food_table": food_table,
            "model": {"backbone": "efficientnet_b0", "image_size": 224},
            "train": {"weights": classifier_weights},
        }
        return cls(detector_config, classifier_config)

    @staticmethod
    def _load_full_table(food_table_path: str) -> Dict[str, Dict[str, Any]]:
        """food_calories.json → english 키로 부가정보(source/kcal_per_100g 등) 매핑."""
        with open(food_table_path, "r", encoding="utf-8") as f:
            raw: List[Dict[str, Any]] = json.load(f)
        return {item["english"]: item for item in raw}

    # ------------------------------------------------------------------
    # 분류 / 박스 처리 헬퍼
    # ------------------------------------------------------------------
    def _classify(self, crop) -> Dict[str, Any]:
        """크롭(numpy RGB) 1장을 분류해 class_id 와 확신도를 반환한다."""
        tensor = self._transform(image=crop)["image"].unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            logits = self.model(tensor)
            probs = logits.softmax(dim=1)
            prob, idx = probs.max(dim=1)
        return {"class_id": int(idx.item()), "class_prob": float(prob.item())}

    @staticmethod
    def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        """두 xyxy 박스의 IoU."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _dedup_boxes(self, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """클래스 무관 IoU>임계 박스는 conf 높은 것만 남긴다(칼로리 이중 합산 방지)."""
        ordered = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
        kept: List[Dict[str, Any]] = []
        for box in ordered:
            if all(self._iou(box["bbox_xyxy"], k["bbox_xyxy"]) <= DEDUP_IOU_THRESHOLD for k in kept):
                kept.append(box)
        return kept

    @staticmethod
    def _pad_and_clip(
        bbox: Tuple[float, float, float, float], W: int, H: int, ratio: float,
    ) -> Tuple[int, int, int, int]:
        """xyxy 박스를 각 변으로 ratio 만큼 넓히고 이미지 경계로 clip 한 정수 좌표를 반환."""
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * ratio, bh * ratio
        nx1 = int(round(max(0, x1 - px)))
        ny1 = int(round(max(0, y1 - py)))
        nx2 = int(round(min(W, x2 + px)))
        ny2 = int(round(min(H, y2 + py)))
        return nx1, ny1, nx2, ny2

    def _make_item(
        self,
        class_id: int,
        class_prob: float,
        bbox_xyxy: Optional[List[float]],
        detect_conf: Optional[float],
        yolo_class: Optional[str],
    ) -> Dict[str, Any]:
        """분류 결과 + 칼로리 정보로 항목 dict 한 개를 만든다."""
        english = self.classes[class_id]
        info = self.eng_to_info[english]
        full = self.eng_to_full.get(english, {})
        return {
            "class_id": class_id,
            "english": english,
            "korean": info.korean,
            "class_prob": round(class_prob, 4),
            "kcal": round(info.kcal_per_serving, 1),
            "grams": round(info.serving_size_g, 1),
            "kcal_per_100g": full.get("kcal_per_100g"),
            "source": full.get("source"),
            "bbox_xyxy": ([round(v, 1) for v in bbox_xyxy] if bbox_xyxy is not None else None),
            "detect_conf": (round(detect_conf, 4) if detect_conf is not None else None),
            "yolo_class": yolo_class,  # 참고용(최종 종류 결정에는 미사용)
        }

    # ------------------------------------------------------------------
    # 메인 추론
    # ------------------------------------------------------------------
    def predict(self, source: Any) -> Dict[str, Any]:
        """단일 이미지에 대해 단일/다중 경로로 음식 종류·칼로리와 총합을 추정한다.

        Returns:
            {
              "mode": "single" | "multi",
              "items": [
                  {class_id, english, korean, class_prob, kcal, grams,
                   kcal_per_100g, source, bbox_xyxy, detect_conf, yolo_class}, ...
              ],
              "total_kcal": float,
              "num_items": int,
              "num_boxes": int,            # 중복 제거 후 검출 박스 수
              "image_shape": (H, W),
            }
        """
        det = self.detector.detect(source)
        orig = det["orig_img"]                  # numpy (H, W, 3), BGR
        H, W = orig.shape[:2]
        boxes = self._dedup_boxes(det["boxes"])

        items: List[Dict[str, Any]] = []
        total_kcal = 0.0

        if len(boxes) >= MULTI_MIN_BOXES:
            mode = "multi"
            for box in boxes:
                cx1, cy1, cx2, cy2 = self._pad_and_clip(box["bbox_xyxy"], W, H, CROP_PADDING_RATIO)
                sub = orig[cy1:cy2, cx1:cx2]
                if sub.size == 0:
                    continue
                crop = sub[:, :, ::-1]          # BGR -> RGB
                cls = self._classify(crop)
                item = self._make_item(
                    class_id=cls["class_id"],
                    class_prob=cls["class_prob"],
                    bbox_xyxy=list(box["bbox_xyxy"]),
                    detect_conf=box["confidence"],
                    yolo_class=box.get("yolo_class"),
                )
                total_kcal += item["kcal"]
                items.append(item)
        else:
            # 박스 0~1개 → 검증된 단일 경로: 전체 이미지를 그대로 분류한다.
            mode = "single"
            crop = orig[:, :, ::-1]             # 전체 이미지 BGR -> RGB
            cls = self._classify(crop)
            detect_conf = boxes[0]["confidence"] if len(boxes) == 1 else None
            yolo_class = boxes[0].get("yolo_class") if len(boxes) == 1 else None
            item = self._make_item(
                class_id=cls["class_id"],
                class_prob=cls["class_prob"],
                bbox_xyxy=None,                  # 전체 이미지 기준이라 박스 없음
                detect_conf=detect_conf,
                yolo_class=yolo_class,
            )
            total_kcal += item["kcal"]
            items.append(item)

        return {
            "mode": mode,
            "items": items,
            "total_kcal": round(total_kcal, 1),
            "num_items": len(items),
            "num_boxes": len(boxes),
            "image_shape": (H, W),
        }

    # ------------------------------------------------------------------
    # 시각화
    # ------------------------------------------------------------------
    def render_image(self, source: Any) -> Tuple[Any, Dict[str, Any]]:
        """추론 후 박스가 그려진 RGB numpy 이미지와 결과 dict 를 함께 반환한다.

        다중 경로: 각 박스에 "분류이름 p확신도 kcal" 라벨 표시.
        단일 경로: 박스가 없으므로 원본 이미지를 그대로 돌려준다.
        """
        import cv2  # 지연 임포트
        import numpy as np

        if isinstance(source, str):
            img_bgr = cv2.imread(source)
        elif isinstance(source, np.ndarray):
            # numpy 입력은 RGB 로 가정(PIL→np 변환과 일치) → BGR 로 맞춰 그린다.
            img_bgr = source[:, :, ::-1].copy()
        else:
            img_bgr = np.array(source)[:, :, ::-1].copy()

        result = self.predict(source)
        for item in result["items"]:
            if item["bbox_xyxy"] is None:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in item["bbox_xyxy"])
            label = f"{item['english']} p{item['class_prob']:.2f} {item['kcal']:.0f}kcal"
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img_bgr, label, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        img_rgb = img_bgr[:, :, ::-1].copy()
        return img_rgb, result

    def render(self, source: Any, out_path: str) -> Dict[str, Any]:
        """추론 결과를 이미지에 그려 out_path 로 저장하고 결과 dict 를 반환한다(CLI 용)."""
        import cv2  # 지연 임포트

        img_rgb, result = self.render_image(source)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, img_rgb[:, :, ::-1])  # RGB -> BGR 로 저장
        return result
