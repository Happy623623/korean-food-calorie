"""학습된 YOLO11 검출기로 음식 영역 박스를 추론한다 (하이브리드 1단계).

종류 분류는 하지 않고 위치(bbox)와 검출 신뢰도만 돌려준다.
원본 이미지를 함께 반환하므로, 파이프라인은 이를 크롭해 classifier 로 넘긴다.

ultralytics 의존성은 검출기를 실제로 생성할 때만 로드된다(지연 임포트).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List


class FoodDetector:
    """YOLO11 기반 음식 영역 검출기."""

    def __init__(self, config: Dict[str, Any]) -> None:
        from ultralytics import YOLO  # 지연 임포트

        m = config["model"]
        weights = m["weights"]
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"검출기 가중치를 찾을 수 없습니다: {weights}\n"
                f"먼저 `python -m src.detector.train_yolo` 로 학습한 뒤 "
                f"best.pt 를 이 경로로 지정/복사하세요."
            )

        self.model = YOLO(weights)
        self.conf = float(m.get("conf", 0.25))
        self.iou = float(m.get("iou", 0.45))
        self.imgsz = int(m.get("imgsz", 640))

    def detect(self, source: Any) -> Dict[str, Any]:
        """단일 이미지에서 음식 영역 박스를 검출한다.

        Args:
            source: 이미지 경로(str) 또는 numpy 배열 등 ultralytics 가 받는 입력

        Returns:
            {
              "boxes": [{"bbox_xyxy": (x1,y1,x2,y2), "confidence": float}, ...],
              "orig_img": numpy 배열 (H, W, 3) BGR,   # 크롭용 원본
              "image_shape": (H, W),
            }
        """
        results = self.model.predict(
            source=source, conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, verbose=False,
        )
        result = results[0]

        boxes: List[Dict[str, Any]] = []
        for box in result.boxes:
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            boxes.append({"bbox_xyxy": (x1, y1, x2, y2), "confidence": confidence})

        return {
            "boxes": boxes,
            "orig_img": result.orig_img,        # numpy (H, W, 3), BGR
            "image_shape": tuple(int(v) for v in result.orig_shape),
        }
