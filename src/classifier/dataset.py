"""classifier(EfficientNet 23-class) 학습용 폴더 기반 Dataset + albumentations 변환.

데이터 레이아웃 (ImageFolder 스타일):

    <data_root>/
        gimbap/      *.jpg
        tteokbokki/  *.jpg
        bibimbap/    *.jpg
        ...

하위 폴더 이름은 food_calories.json 의 영문명(english)과 일치시킨다.
클래스 인덱스는 인자로 받은 `classes`(food_calories.json 순서) 를 그대로 따르므로,
torchvision.ImageFolder 의 알파벳 정렬과 달리 칼로리 테이블과 정렬이 항상 일치한다.
"""

import os
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet 사전학습 정규화 통계값 (EfficientNet 사전학습 기준)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_transforms(image_size: int, train: bool) -> A.Compose:
    """학습/평가용 albumentations 변환 파이프라인을 생성한다.

    Args:
        image_size: 정사각형 입력 크기 (예: 224)
        train: True면 증강 포함, False면 리사이즈 + 정규화만 적용
    """
    if train:
        # Colab 학습 레시피와 동일(torchvision 등가):
        #   RandomResizedCrop(224, scale 0.7~1.0) + HorizontalFlip
        #   + Rotation(±15) + ColorJitter(0.2, 0.2, 0.2, 0.05)
        # Rotate/ColorJitter 는 torchvision 처럼 매 호출 적용(p=1.0), Flip 만 p=0.5.
        return A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    # 평가/추론: 학습과 동일하게 짧은 변 256 으로 리사이즈(비율 유지) 후 224 중앙 크롭.
    # torchvision 의 Resize(256)→CenterCrop(224) 와 동치.
    resize = int(round(image_size / 0.875))  # 224 → 256
    return A.Compose([
        A.SmallestMaxSize(max_size=resize),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class FoodClassificationDataset(Dataset):
    """폴더 기반 음식 분류 데이터셋.

    클래스 인덱스는 `classes` 인자의 순서를 그대로 사용한다.
    데이터가 아직 없어도(준비 전이어도) 인스턴스 생성은 에러 없이 동작하며,
    이때 len(dataset) == 0 이 된다.
    """

    def __init__(
        self,
        data_root: str,
        classes: Sequence[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.data_root = data_root
        self.classes: List[str] = list(classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.transform = transform
        self.samples: List[Tuple[str, int]] = self._scan()

    def _scan(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for name in self.classes:
            cls_dir = os.path.join(self.data_root, name)
            if not os.path.isdir(cls_dir):
                # 해당 클래스 폴더가 아직 없으면 건너뜀 (에러 X)
                continue
            label = self.class_to_idx[name]
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    samples.append((os.path.join(cls_dir, fname), label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


def create_classification_dataloader(
    data_root: str,
    classes: Sequence[str],
    image_size: int,
    batch_size: int,
    train: bool,
    num_workers: int = 4,
) -> DataLoader:
    """FoodClassificationDataset 으로부터 DataLoader 를 생성한다."""
    dataset = FoodClassificationDataset(
        data_root=data_root,
        classes=classes,
        transform=build_transforms(image_size, train=train),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
