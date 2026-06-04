"""classifier 모델: torchvision EfficientNet-B0 기반 음식 분류기 (23-class).

하이브리드 파이프라인 2단계. YOLO 가 잘라준 음식 크롭을 입력받아
음식 종류 로짓을 출력한다. 출력은 분류용이므로 활성화 없이 로짓 그대로 반환하며,
손실은 학습 코드에서 CrossEntropy 로 계산한다.

체크포인트 호환:
    build_classifier() 는 torchvision efficientnet_b0 모델을 **그대로**(래퍼 없이) 반환한다.
    따라서 state_dict 키가 `features.*`, `classifier.1.*` 형태가 되어,
    Colab 등에서 `efficientnet_b0(...); model.classifier[1] = Linear(in, N)` 로 학습해
    `{"state_dict": model.state_dict(), "classes": [...]}` 로 저장한 체크포인트를
    접두사 변환 없이 바로 로드할 수 있다.
"""

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_classifier(
    backbone: str = "efficientnet_b0",
    num_classes: int = 23,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """분류기를 생성한다 (torchvision efficientnet_b0).

    classifier head 는 기본 구조(Sequential(Dropout, Linear)) 를 유지하되
    마지막 Linear 를 num_classes 출력으로 교체한다(= classifier[1]).

    학습(train.py)은 pretrained=True + dropout 으로, 추론(inference/pipeline)은
    pretrained=False(체크포인트로 덮어씀) + dropout 무관 으로 호출한다.

    Args:
        backbone: 백본 이름. 현재 빌더는 efficientnet_b0 만 지원한다.
        num_classes: 출력 클래스 수.
        pretrained: True 면 ImageNet 사전학습 가중치로 초기화.
        dropout: classifier head 의 Dropout 확률(구조 변화 없음, 파라미터 없음).
    """
    if backbone != "efficientnet_b0":
        raise ValueError(
            f"이 빌더는 torchvision efficientnet_b0 만 지원합니다 (요청: {backbone!r}). "
            "다른 백본이 필요하면 model.py 를 확장하세요."
        )

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    # torchvision efficientnet_b0 의 classifier = Sequential(Dropout(p=0.2), Linear(1280, 1000))
    # → 마지막 Linear 만 num_classes 출력으로 교체(= classifier[1]).
    in_features = model.classifier[1].in_features
    if dropout is not None:
        model.classifier[0].p = float(dropout)   # 구조/키 변화 없이 확률만 조정
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
