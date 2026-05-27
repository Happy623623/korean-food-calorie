"""공용 헬퍼 모음.

시드 고정, 설정 로드, 로깅, 메트릭, 체크포인트 입출력 등
여러 스크립트에서 공통으로 쓰는 유틸리티를 모아둔다.
"""

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """재현성을 위해 모든 난수 생성기의 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN 결정론적 동작(성능과 트레이드오프)
    torch.backends.cudnn.benchmark = True


def load_config(path: str) -> Dict[str, Any]:
    """YAML 설정 파일을 읽어 dict로 반환한다."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """사용 가능한 디바이스(GPU 우선)를 반환한다."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str = "korean-food", log_dir: Optional[str] = None) -> logging.Logger:
    """콘솔 + (선택) 파일 핸들러를 가진 로거를 생성한다.

    Args:
        name: 로거 이름
        log_dir: 지정 시 해당 폴더에 train.log 파일로도 기록
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # 중복 핸들러 등록 방지
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "train.log"), encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """손실/정확도 등의 누적 평균을 추적하는 헬퍼."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0   # 가장 최근 값
        self.sum = 0.0   # 누적 합
        self.count = 0   # 누적 샘플 수
        self.avg = 0.0   # 현재까지의 평균

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """top-1 정확도를 계산한다.

    Args:
        outputs: 모델 로짓, shape (B, num_classes)
        targets: 정답 레이블, shape (B,)
    """
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(targets.size(0), 1)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """체크포인트 dict를 path에 저장한다(상위 폴더 자동 생성)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


# ======================================================================
# 음식 칼로리 테이블 (클래스/칼로리의 단일 진실 공급원)
# ======================================================================
@dataclass
class FoodInfo:
    """음식 1종의 기준 정보 (food_calories.json 한 항목)."""

    korean: str
    english: str
    kcal_per_serving: float
    serving_size_g: float


class FoodTable:
    """클래스 인덱스 -> FoodInfo 매핑 테이블.

    food_calories.json 의 항목 순서가 곧 클래스 인덱스(0~N-1)이며,
    이는 분류기 학습 시 폴더->인덱스 매핑, 추론 시 인덱스->칼로리 매핑의
    단일 진실 공급원이 된다.
    """

    def __init__(self, infos: Sequence[FoodInfo]) -> None:
        self.infos: List[FoodInfo] = list(infos)

    @classmethod
    def from_json(cls, json_path: str) -> "FoodTable":
        """food_calories.json 을 읽어 테이블을 만든다."""
        with open(json_path, "r", encoding="utf-8") as f:
            raw: List[Dict[str, Any]] = json.load(f)
        infos = [
            FoodInfo(
                korean=item["korean"],
                english=item["english"],
                kcal_per_serving=float(item["kcal_per_serving"]),
                serving_size_g=float(item["serving_size_g"]),
            )
            for item in raw
        ]
        return cls(infos)

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, class_id: int) -> FoodInfo:
        return self.infos[class_id]

    @property
    def korean_names(self) -> List[str]:
        """클래스 인덱스 순서대로의 한글 이름 목록."""
        return [info.korean for info in self.infos]

    @property
    def english_names(self) -> List[str]:
        """클래스 인덱스 순서대로의 영문 이름 목록(= 분류 데이터 폴더 이름)."""
        return [info.english for info in self.infos]

    def kcal(self, class_id: int, servings: float = 1.0) -> float:
        """주어진 클래스/인분 수에 대한 칼로리(kcal)를 계산한다."""
        return self.infos[class_id].kcal_per_serving * servings

    def grams(self, class_id: int, servings: float = 1.0) -> float:
        """주어진 클래스/인분 수에 대한 추정 중량(g)을 계산한다."""
        return self.infos[class_id].serving_size_g * servings
