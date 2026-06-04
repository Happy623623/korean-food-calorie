"""classifier 학습: 음식 크롭 -> 23종 분류 (torchvision EfficientNet-B0).

사용법 (프로젝트 루트에서):
    python -m src.classifier.train --config configs/classifier.yaml

동작:
    - food_calories.json 순서로 클래스(영문명)를 정해 인덱스 정렬을 보장
    - data_root/<split>/<english_name>/*.jpg 폴더 구조에서 학습/검증 데이터를 읽음
    - clamp(max=3) 한 balanced 클래스 가중치 + label smoothing 으로 클래스 불균형 보정
    - AdamW + CosineAnnealingLR + AMP(mixed precision) 로 학습
    - 검증 top-1 정확도가 가장 높은 시점의 가중치를 best.pt 로 저장
      (체크포인트에 classes / backbone / image_size 메타데이터를 함께 기록해
       추론 파이프라인이 설정 없이도 클래스 정렬을 복원할 수 있게 한다)

⚠️ 학습 데이터가 준비되어 있어야 한다. 폴더가 비어 있으면 안내 후 종료한다.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from .dataset import create_classification_dataloader
from .model import build_classifier
from ..utils import (
    AverageMeter,
    FoodTable,
    accuracy,
    get_device,
    get_logger,
    load_config,
    save_checkpoint,
    set_seed,
)


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """한 epoch 학습 또는 검증. optimizer 가 None 이면 평가 모드."""
    train = optimizer is not None
    model.train(train)
    loss_meter, acc_meter = AverageMeter(), AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with autocast("cuda", enabled=scaler is not None and scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy(logits, labels), bs)

    return loss_meter.avg, acc_meter.avg


def _build_class_weights(path, classes, clamp_max, device, logger):
    """클래스 가중치 텐서를 만든다.

    configs/class_weights.json 은 train split 역빈도(balanced) **원시값**을 담는다.
    여기서 classes(=food_calories.json 순서)대로 정렬하고 clamp_max 로 상한을 걸어
    소수 클래스 과대 가중을 막는다(Colab 학습 레시피와 동일: max=3.0).
    파일이 없으면 None 을 반환해 클래스 가중치 없이(균등) 학습한다.
    """
    if not path or not os.path.isfile(path):
        logger.warning("class_weights 파일 없음(%s) -> 클래스 가중치 없이 학습", path)
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    weights = [min(float(raw.get(name, 1.0)), clamp_max) for name in classes]
    logger.info("클래스 가중치 적용: clamp(max=%.1f) · %d개 클래스", clamp_max, len(weights))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="음식 23종 분류기 학습 (EfficientNet-B0)")
    parser.add_argument("--config", default="configs/classifier.yaml", help="설정 파일 경로")
    parser.add_argument("--epochs", type=int, default=None, help="config 값 덮어쓰기")
    parser.add_argument("--batch", type=int, default=None, help="config 값 덮어쓰기")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("project", {}).get("seed", 42))

    data_cfg = config["data"]
    model_cfg = config["model"]
    tr = config["train"]

    # 클래스/칼로리의 단일 진실 공급원: food_calories.json 순서 = 클래스 인덱스
    table = FoodTable.from_json(config["food_table"])
    classes = table.english_names          # 폴더 이름과 일치하는 영문명
    num_classes = len(classes)

    image_size = int(model_cfg.get("image_size", 224))
    batch_size = int(args.batch if args.batch is not None else tr["batch_size"])
    epochs = int(args.epochs if args.epochs is not None else tr["epochs"])
    num_workers = int(data_cfg.get("num_workers", 4))

    device = get_device()
    logger = get_logger(log_dir=tr.get("log_dir"))
    logger.info("device=%s | classes=%d | epochs=%d | batch=%d",
                device, num_classes, epochs, batch_size)

    train_loader = create_classification_dataloader(
        os.path.join(data_cfg["root"], data_cfg["train"]),
        classes, image_size, batch_size, train=True, num_workers=num_workers,
    )
    val_loader = create_classification_dataloader(
        os.path.join(data_cfg["root"], data_cfg["val"]),
        classes, image_size, batch_size, train=False, num_workers=num_workers,
    )

    if len(train_loader.dataset) == 0:
        logger.error(
            "학습 데이터가 비어 있습니다(%s). "
            "data_root/<split>/<english_name>/*.jpg 구조로 이미지를 먼저 준비하세요.",
            os.path.join(data_cfg["root"], data_cfg["train"]),
        )
        return
    logger.info("train=%d장 | val=%d장",
                len(train_loader.dataset), len(val_loader.dataset))

    model = build_classifier(
        backbone=model_cfg.get("backbone", "efficientnet_b0"),
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.2),
    ).to(device)

    class_weights = _build_class_weights(
        tr.get("class_weights", os.path.join("configs", "class_weights.json")),
        classes, float(tr.get("class_weight_clamp", 3.0)), device, logger,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=float(tr.get("label_smoothing", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = bool(tr.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    best_acc = 0.0
    ckpt_dir = tr.get("checkpoint_dir", "checkpoints")
    best_path = tr.get("weights", os.path.join(ckpt_dir, "classifier_best.pt"))

    def make_state(epoch):
        return {
            "model": model.state_dict(),
            "epoch": epoch,
            "classes": classes,
            "num_classes": num_classes,
            "backbone": model_cfg.get("backbone", "efficientnet_b0"),
            "image_size": image_size,
        }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler,
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "epoch %d/%d | lr %.2e | train_loss %.4f acc %.3f | val_loss %.4f acc %.3f",
            epoch, epochs, scheduler.get_last_lr()[0],
            train_loss, train_acc, val_loss, val_acc,
        )

        save_checkpoint(make_state(epoch), os.path.join(ckpt_dir, "classifier_last.pt"))
        if len(val_loader.dataset) > 0 and val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(make_state(epoch), best_path)
            logger.info("  best 갱신 -> %s (val_acc %.3f)", best_path, best_acc)

    logger.info("학습 종료. best val_acc=%.3f -> %s", best_acc, best_path)


if __name__ == "__main__":
    main()
