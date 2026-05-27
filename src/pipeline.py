"""하이브리드 추론 파이프라인.

    이미지 ──▶ [1단계: YOLO11 검출] ──▶ 음식 영역 bbox
                                        │
                                        ▼  각 bbox 크롭
            [2단계: EfficientNet 분류] ──▶ 음식 종류(15종) + 확신도
                                        │
                                        ▼  검출 박스 1개 = 1인분
                       kcal = food_calories.json 의 kcal_per_serving 합산

portion 회귀는 하지 않는다. 검출된 박스 1개당 해당 음식의 "1인분 평균 kcal" 로 합산한다.
torch / ultralytics / opencv 의존성은 파이프라인을 실제로 생성할 때만 로드된다(지연 임포트).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from .utils import FoodTable, get_device


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

        # --- 1단계: 검출기 ---
        self.detector = FoodDetector(detector_config)

        # --- 칼로리/클래스 단일 진실 공급원 ---
        self.table = FoodTable.from_json(classifier_config["food_table"])

        # --- 2단계: 분류기 ---
        cm = classifier_config["model"]
        weights = classifier_config["train"]["weights"]
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"분류기 가중치를 찾을 수 없습니다: {weights}\n"
                f"먼저 `python -m src.classifier.train` 로 학습하세요."
            )

        checkpoint = torch.load(weights, map_location=self.device)
        # 체크포인트에 저장된 메타데이터로 클래스 정렬/입력 크기를 복원
        self.classes: List[str] = checkpoint.get("classes", self.table.english_names)
        backbone = checkpoint.get("backbone", cm.get("backbone", "efficientnet_b0"))
        self.image_size = int(checkpoint.get("image_size", cm.get("image_size", 224)))
        num_classes = int(checkpoint.get("num_classes", len(self.classes)))

        # 정합성 검증: 분류기 클래스 순서가 칼로리 테이블과 어긋나면 즉시 에러
        if self.classes != self.table.english_names:
            raise ValueError(
                "분류기 체크포인트의 클래스 순서가 food_calories.json 과 일치하지 않습니다.\n"
                f"  checkpoint: {self.classes}\n"
                f"  food_table: {self.table.english_names}\n"
                "동일한 food_calories.json 으로 분류기를 다시 학습하세요."
            )

        self.model = build_classifier(
            backbone=backbone, num_classes=num_classes, pretrained=False, dropout=0.0,
        )
        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self._transform = build_transforms(self.image_size, train=False)

    def _classify(self, crop) -> Dict[str, Any]:
        """크롭(numpy RGB) 1장을 분류해 class_id 와 확신도를 반환한다."""
        tensor = self._transform(image=crop)["image"].unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            logits = self.model(tensor)
            probs = logits.softmax(dim=1)
            prob, idx = probs.max(dim=1)
        return {"class_id": int(idx.item()), "class_prob": float(prob.item())}

    def predict(self, source: Any) -> Dict[str, Any]:
        """단일 이미지에 대해 음식별 종류/칼로리와 총합을 추정한다.

        Returns:
            {
              "detections": [
                  {class_id, name, english, detect_conf, class_prob, bbox_xyxy, kcal, grams}, ...
              ],
              "total_kcal": float,
              "num_items": int,
              "image_shape": (h, w),
            }
        """
        det = self.detector.detect(source)
        orig = det["orig_img"]                  # numpy (H, W, 3), BGR

        detections: List[Dict[str, Any]] = []
        total_kcal = 0.0

        for box in det["boxes"]:
            x1, y1, x2, y2 = box["bbox_xyxy"]
            ix1, iy1, ix2, iy2 = (int(round(v)) for v in (x1, y1, x2, y2))
            sub = orig[max(0, iy1):iy2, max(0, ix1):ix2]
            if sub.size == 0:
                continue
            crop = sub[:, :, ::-1]              # BGR -> RGB

            cls = self._classify(crop)
            class_id = cls["class_id"]
            info = self.table[class_id]

            # 검출 박스 1개 = 표준 1인분
            kcal = info.kcal_per_serving
            grams = info.serving_size_g
            total_kcal += kcal

            detections.append({
                "class_id": class_id,
                "name": info.korean,
                "english": info.english,
                "detect_conf": round(box["confidence"], 4),
                "class_prob": round(cls["class_prob"], 4),
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "kcal": round(kcal, 1),
                "grams": round(grams, 1),
            })

        return {
            "detections": detections,
            "total_kcal": round(total_kcal, 1),
            "num_items": len(detections),
            "image_shape": det["image_shape"],
        }

    def render(self, source: Any, out_path: str) -> Dict[str, Any]:
        """추론 결과를 이미지에 그려 out_path 로 저장하고, 결과 dict 를 반환한다.

        각 박스에 "영문이름 p확신도 (kcal)" 라벨을 표시한다.
        (한글은 OpenCV 기본 폰트로 깨질 수 있어 영문/숫자 위주로 표기)
        """
        import cv2  # 지연 임포트

        img = cv2.imread(source) if isinstance(source, str) else source.copy()
        result = self.predict(source)

        for det in result["detections"]:
            x1, y1, x2, y2 = (int(round(v)) for v in det["bbox_xyxy"])
            label = f"{det['english']} p{det['class_prob']:.2f} {det['kcal']:.0f}kcal"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, img)
        return result
