"""classifier 평가: test 셋에 대해 정확도/리포트/혼동행렬을 산출하고 저장한다.

사용법 (프로젝트 루트에서):
    python -m src.classifier.evaluate
    python -m src.classifier.evaluate --log-file logs/train_20260531_162241.log

동작:
    - checkpoints/classifier_best.pt 를 로드 (classes/backbone/image_size 메타데이터 사용)
    - data/classifier/test/<english_name>/*.jpg 를 평가
    - evaluation_results/ 에 리포트(txt/json) + 혼동행렬(png/csv) + (선택)학습곡선(png) 저장

⚠️ 학습 데이터(test 폴더)와 체크포인트가 준비되어 있어야 한다.
"""

import argparse
import json
import os
import re

# GUI 없는 환경(서버/CI)에서도 PNG 저장이 가능하도록 비대화형 백엔드를 강제한다.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .dataset import create_classification_dataloader  # noqa: E402
from .model import build_classifier  # noqa: E402
from ..utils import get_device  # noqa: E402


# 체크포인트마다 state_dict 키 이름이 다를 수 있어 우선순위대로 탐색한다.
_STATE_DICT_KEYS = ("model", "model_state_dict", "state_dict")


def load_state_dict_into(model, ckpt) -> None:
    """체크포인트 dict 에서 모델 가중치 키를 찾아 model 에 로드한다.

    train.py 는 'model' 키에 저장하지만, 다른 코드/버전에서 만든 체크포인트도
    받을 수 있도록 흔한 키들을 순서대로 탐색한다. 하나도 없으면 키 목록과 함께 실패.
    """
    for k in _STATE_DICT_KEYS:
        if k in ckpt:
            model.load_state_dict(ckpt[k])
            break
    else:
        raise KeyError(f"체크포인트에 모델 가중치 키 없음. 키 목록: {list(ckpt.keys())}")


@torch.no_grad()
def run_inference(model, loader, device):
    """전체 test 셋을 추론해 (y_true, y_pred) numpy 배열을 반환한다."""
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def parse_training_log(log_file: str):
    """train.log 에서 epoch 별 train/val loss·acc 를 정규식으로 파싱한다.

    예시 라인:
        "[2026-05-31 16:18:18] INFO | epoch 27/30 | lr 7.34e-06 |
         train_loss 0.5799 acc 0.999 | val_loss 0.7607 acc 0.929"
    """
    pattern = re.compile(
        r"epoch\s+(?P<epoch>\d+)\s*/\s*\d+.*?"
        r"train_loss\s+(?P<train_loss>[\d.]+)\s+acc\s+(?P<train_acc>[\d.]+).*?"
        r"val_loss\s+(?P<val_loss>[\d.]+)\s+acc\s+(?P<val_acc>[\d.]+)"
    )
    rows = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({
                    "epoch": int(m.group("epoch")),
                    "train_loss": float(m.group("train_loss")),
                    "train_acc": float(m.group("train_acc")),
                    "val_loss": float(m.group("val_loss")),
                    "val_acc": float(m.group("val_acc")),
                })
    return rows


def plot_training_curves(rows, out_path: str) -> None:
    """학습 곡선(loss/accuracy)을 2x1 subplot 으로 저장한다."""
    epochs = [r["epoch"] for r in rows]
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    ax_loss.plot(epochs, [r["train_loss"] for r in rows], "o-", label="train_loss")
    ax_loss.plot(epochs, [r["val_loss"] for r in rows], "s-", label="val_loss")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training / Validation Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, [r["train_acc"] for r in rows], "o-", label="train_acc")
    ax_acc.plot(epochs, [r["val_acc"] for r in rows], "s-", label="val_acc")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Training / Validation Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, classes, out_path: str) -> None:
    """혼동행렬을 1x2(개수 / 행 정규화) seaborn heatmap 으로 저장한다."""
    # 행 정규화: 각 실제 클래스(행) 합으로 나눠 비율로 표현. 0으로 나눔 방지.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float),
                        where=row_sums != 0)

    side = max(8, len(classes))
    fig, (ax_cnt, ax_norm) = plt.subplots(1, 2, figsize=(2 * side, side))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax_cnt)
    ax_cnt.set_title("Confusion Matrix (counts)")
    ax_cnt.set_xlabel("predicted")
    ax_cnt.set_ylabel("true")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax_norm)
    ax_norm.set_title("Confusion Matrix (row-normalized)")
    ax_norm.set_xlabel("predicted")
    ax_norm.set_ylabel("true")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def top_confused_pairs(cm, classes, top_k: int = 5):
    """혼동행렬 비대각 원소 중 큰 순으로 (true, pred, count) 를 반환한다."""
    pairs = []
    n = len(classes)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((classes[i], classes[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="음식 23종 분류기 test 셋 평가")
    parser.add_argument("--checkpoint", default="checkpoints/best_efficientnet_b0_23.pt",
                        help="평가할 체크포인트 경로")
    parser.add_argument("--data-root", default="data/classifier/test",
                        help="test 데이터 루트 (하위에 <english_name>/*.jpg)")
    parser.add_argument("--batch", type=int, default=32, help="배치 크기")
    parser.add_argument("--output-dir", default="evaluation_results",
                        help="평가 결과 저장 폴더")
    parser.add_argument("--log-file", default=None,
                        help="train.log 경로. 주면 학습 곡선 PNG도 생성")
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 체크포인트 로드 + 메타데이터 (classes 등 비-텐서 포함 → weights_only=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt.get("classes")
    if not classes:
        raise ValueError(
            f"체크포인트({args.checkpoint})에 classes 메타데이터가 없습니다. "
            "train.py 로 저장한 체크포인트인지 확인하세요."
        )
    backbone = ckpt.get("backbone", "efficientnet_b0")
    image_size = int(ckpt.get("image_size", 224))
    num_classes = len(classes)

    print(f"device       : {device}")
    print(f"checkpoint   : {args.checkpoint} (epoch={ckpt.get('epoch')})")
    print(f"backbone     : {backbone} | image_size={image_size}")
    print(f"classes      : {num_classes}개")

    # 2) 모델 빌드 (추론용: pretrained=False, dropout=0) + 가중치 로드
    model = build_classifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=False,
        dropout=0.0,
    ).to(device)
    load_state_dict_into(model, ckpt)

    # 3) DataLoader (평가용: train=False → 증강/셔플 없음)
    loader = create_classification_dataloader(
        data_root=args.data_root,
        classes=classes,
        image_size=image_size,
        batch_size=args.batch,
        train=False,
    )
    n_samples = len(loader.dataset)
    if n_samples == 0:
        raise ValueError(
            f"test 데이터가 비어 있습니다({args.data_root}). "
            "<data_root>/<english_name>/*.jpg 구조를 확인하세요."
        )
    print(f"test 이미지  : {n_samples}장\n")

    # 4) 추론 + 메트릭
    y_true, y_pred = run_inference(model, loader, device)
    labels_idx = list(range(num_classes))

    test_acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=labels_idx, target_names=classes, digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, zero_division=0,
    )

    # 5) 출력 파일 저장
    saved = []

    report_path = os.path.join(args.output_dir, "test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"test_accuracy: {test_acc:.4f}\n")
        f.write(f"test_samples : {n_samples}\n\n")
        f.write(report)
    saved.append(report_path)

    metrics = {
        "test_accuracy": float(test_acc),
        "test_samples": int(n_samples),
        "per_class": {
            classes[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(num_classes)
        },
    }
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    saved.append(metrics_path)

    cm_png = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes, cm_png)
    saved.append(cm_png)

    cm_csv = os.path.join(args.output_dir, "confusion_matrix.csv")
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(cm_csv, encoding="utf-8-sig")
    saved.append(cm_csv)

    if args.log_file:
        rows = parse_training_log(args.log_file)
        if rows:
            curves_png = os.path.join(args.output_dir, "training_curves.png")
            plot_training_curves(rows, curves_png)
            saved.append(curves_png)
        else:
            print(f"⚠️ 로그 파일({args.log_file})에서 epoch 라인을 찾지 못해 곡선은 건너뜁니다.")

    # 6) 콘솔 출력
    print(f"=== Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%) ===\n")
    print(report)

    confused = top_confused_pairs(cm, classes, top_k=5)
    print("\nTop 5 confused pairs (true → pred : count):")
    if confused:
        for true_c, pred_c, cnt in confused:
            print(f"  {true_c} → {pred_c} : {cnt}")
    else:
        print("  (혼동된 쌍 없음 — 완벽 분류)")

    print("\n저장된 파일:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
