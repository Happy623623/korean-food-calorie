"""classifier 모델: timm EfficientNet-B0 기반 15-class 음식 분류기.

하이브리드 파이프라인 2단계. YOLO 가 잘라준 음식 크롭을 입력받아
음식 종류(15종) 로짓을 출력한다. 출력은 분류용이므로 활성화 없이 로짓 그대로 반환하며,
손실은 학습 코드에서 CrossEntropy 로 계산한다.
"""

import timm
import torch
import torch.nn as nn


class FoodClassifier(nn.Module):
    """EfficientNet 백본 + num_classes-출력 분류 head."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # num_classes 를 주면 timm 이 백본 위에 분류 head 를 붙인다.
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)            # shape (B, num_classes) 로짓


def build_classifier(
    backbone: str = "efficientnet_b0",
    num_classes: int = 15,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> FoodClassifier:
    """분류기를 생성한다.

    학습(train.py)은 사전학습 가중치 + dropout 으로, 추론(pipeline.py)은
    pretrained=False(체크포인트로 덮어씀) + dropout=0 으로 호출한다.
    """
    return FoodClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
