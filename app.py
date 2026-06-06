"""한식 칼로리 분석기 Gradio 데모 (하이브리드: 단일/다중 검출).

한식 사진 한 장을 받아:
  - 음식이 1개(또는 검출 0~1개)면  → 기존 단일 분류 UI(종류 + 1인분 kcal + Top-3)를 유지.
  - 음식이 여러 개(검출 ≥2개)면     → 박스 시각화 + 항목별 표 + 총 칼로리를 보여준다.

1단계 YOLO11(checkpoints/yolo11n_23cls.pt, conf=0.35, CPU)로 음식 영역을 찾고,
2단계 EfficientNet-B0 분류기가 각 영역의 종류를 결정한다(YOLO 종류는 참고 표시).

실행 (프로젝트 루트에서):
    python app.py
    → http://127.0.0.1:7860 접속
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import gradio as gr

from src.classifier.inference import FoodClassifier

logger = logging.getLogger("korean-food.app")
logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
CLASSIFIER_CKPT = "checkpoints/best_efficientnet_b0_23.pt"
DETECTOR_CKPT = "checkpoints/yolo11n_23cls.pt"
FOOD_TABLE_PATH = "food_calories.json"
EXAMPLES_ROOT = "examples"                  # 배포용 예제(저작권 안전 이미지) 우선
TEST_ROOT = "data/classifier/test"          # 폴백: 로컬 데이터셋
LOW_CONFIDENCE_THRESHOLD = 0.50             # 이 미만이면 경고 표시
DETECTOR_CONF = 0.35                        # YOLO 검출 신뢰도 임계값
GITHUB_URL = "https://github.com/Happy623623/korean-food-calorie"

# 칼로리 출처 코드 → 사람이 읽기 좋은 한글 라벨
SOURCE_LABELS = {
    "MFDS": "식약처(MFDS) 공식 데이터",
    "estimate": "추정값(estimate)",
}

# ----------------------------------------------------------------------
# 모델 로드 (전역 1회)
# ----------------------------------------------------------------------
# 단일 경로용 분류기(기존 동작 그대로: Top-3 포함).
try:
    CLASSIFIER: Optional[FoodClassifier] = FoodClassifier(
        checkpoint_path=CLASSIFIER_CKPT,
        food_table_path=FOOD_TABLE_PATH,
        device="cpu",
    )
    SUPPORTED_KOREAN: List[str] = [
        CLASSIFIER.food_table.get(c, {}).get("korean") or c
        for c in CLASSIFIER.classes
    ]
except Exception as exc:
    logger.error("FoodClassifier 로드 실패: %s", exc)
    CLASSIFIER = None
    SUPPORTED_KOREAN = []

# 하이브리드 파이프라인(검출 → 분류 → 칼로리). 검출기/의존성 누락 시 None → 단일 경로만 사용.
try:
    from src.pipeline import FoodCaloriePipeline

    PIPELINE: Optional["FoodCaloriePipeline"] = FoodCaloriePipeline.from_paths(
        detector_weights=DETECTOR_CKPT,
        classifier_weights=CLASSIFIER_CKPT,
        food_table=FOOD_TABLE_PATH,
        conf=DETECTOR_CONF,
        infer_device="cpu",
    )
    logger.info("하이브리드 파이프라인 로드 완료(다중 검출 지원)")
except Exception as exc:
    logger.warning("파이프라인 로드 실패 → 단일 분류 경로만 사용: %s", exc)
    PIPELINE = None


# ----------------------------------------------------------------------
# 헬퍼
# ----------------------------------------------------------------------
def _source_label(source: Optional[str]) -> str:
    """출처 코드를 한글 라벨로 변환한다(미지정/미상은 그대로)."""
    if not source:
        return "출처 미상"
    return SOURCE_LABELS.get(source, source)


def _single_outputs(image) -> Tuple[str, Dict[str, float], str, str]:
    """단일 경로(기존 동작): 전체 이미지를 분류해 4개 출력을 만든다."""
    result = CLASSIFIER.predict(image, top_k=3)
    top1 = result["top1"]
    korean = top1["korean"] or "알 수 없음"
    english = top1["english"]
    confidence = top1["confidence"]

    lines = [f"## 🍱 {korean} ({english})", f"신뢰도 **{confidence * 100:.1f}%**", ""]
    kcal = top1["kcal_per_serving"]
    grams = top1["serving_size_g"]
    if kcal is not None:
        grams_str = f" ({grams:g}g)" if grams is not None else ""
        lines.append(f"**1인분 {kcal:g} kcal**{grams_str}")
        if top1["kcal_per_100g"] is not None:
            lines.append(
                f"100g당 {top1['kcal_per_100g']:g} kcal "
                f"(출처: {_source_label(top1['source'])})"
            )
    else:
        lines.append("_이 음식의 칼로리 정보가 테이블에 없습니다._")
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        lines.append("")
        lines.append("> ⚠️ 모델 신뢰도가 낮습니다. 학습된 23종 한식이 아닐 수 있습니다.")
    main_md = "\n".join(lines)

    top_k_labels: Dict[str, float] = {}
    for entry in result["top_k"]:
        label = entry["korean"] or entry["english"]
        top_k_labels[label] = float(entry["confidence"])

    if kcal is not None:
        source_md = (
            f"**칼로리 출처:** {_source_label(top1['source'])}\n\n"
            "식약처 데이터는 공식 영양성분 DB 기준이며, "
            "`추정값`은 조리 방식·1인분량을 반영해 보정한 값입니다."
        )
    else:
        source_md = "**칼로리 출처:** 정보 없음"

    debug_md = f"`english_class = {english}`  ·  mode = single"
    return main_md, top_k_labels, source_md, debug_md


def _multi_outputs(image) -> Tuple[object, List[List[object]], str]:
    """다중 경로: 박스 시각화 이미지 + 항목 표(rows) + 총 칼로리 마크다운."""
    img_rgb, result = PIPELINE.render_image(image)

    rows: List[List[object]] = []
    for i, it in enumerate(result["items"], start=1):
        rows.append([
            i,
            it["korean"] or "?",
            it["english"],
            f"{it['class_prob'] * 100:.1f}%",
            f"{it['kcal']:g}",
            it.get("yolo_class") or "-",
        ])

    total_md = (
        f"## 🍽️ 총 {result['total_kcal']:g} kcal\n"
        f"검출 음식 **{result['num_items']}개** "
        f"(검출 박스 {result['num_boxes']}개, 중복 제거 후)\n\n"
        "> 항목별 1인분 평균 칼로리의 합계입니다. 최종 종류는 분류기 판단이며, "
        "표의 *YOLO참고* 열은 검출기가 본 종류(참고용)입니다."
    )
    return img_rgb, rows, total_md


# ----------------------------------------------------------------------
# 추론 콜백 (단일/다중 분기)
# ----------------------------------------------------------------------
# 출력 순서:
#   single_group, main_out, topk_out, source_out, debug_out,
#   multi_group, multi_image, multi_table, multi_total
def predict_food(image):
    show_single = gr.update(visible=True)
    hide = gr.update(visible=False)
    empty_single = ("", {}, "", "")
    empty_multi = (None, [], "")

    # 안전장치: 모델 미로드
    if CLASSIFIER is None and PIPELINE is None:
        msg = (
            "### ⚠️ 모델을 불러오지 못했습니다\n"
            f"`{CLASSIFIER_CKPT}` / `{DETECTOR_CKPT}` 와 `{FOOD_TABLE_PATH}` 를 확인하세요."
        )
        return (show_single, msg, {}, "", "", hide, *empty_multi)

    if image is None:
        msg = "### 👆 한식 사진을 업로드하면 분석 결과가 여기에 표시됩니다."
        return (show_single, msg, {}, "", "", hide, *empty_multi)

    # 1) 파이프라인으로 단일/다중 판단
    mode = "single"
    if PIPELINE is not None:
        try:
            probe = PIPELINE.predict(image)
            mode = probe["mode"]
        except Exception:
            logger.exception("파이프라인 추론 오류 → 단일 경로로 폴백")
            mode = "single"

    # 2) 다중 경로
    if mode == "multi":
        try:
            img_rgb, rows, total_md = _multi_outputs(image)
            # single_group 숨김, multi_group 표시
            return (hide, "", {}, "", "", gr.update(visible=True),
                    img_rgb, rows, total_md)
        except Exception as exc:
            logger.exception("다중 경로 처리 오류 → 단일 경로로 폴백")
            # 폴백: 단일 경로로 진행

    # 3) 단일 경로 (기존 UI 유지)
    if CLASSIFIER is None:
        msg = "### ⚠️ 단일 분류기를 불러오지 못했습니다."
        return (show_single, msg, {}, "", "", hide, *empty_multi)
    try:
        main_md, topk, source_md, debug_md = _single_outputs(image)
    except Exception as exc:
        logger.exception("단일 추론 오류")
        msg = f"### ❌ 분석 중 오류가 발생했습니다\n```\n{exc}\n```"
        return (show_single, msg, {}, "", "", hide, *empty_multi)
    return (show_single, main_md, topk, source_md, debug_md, hide, *empty_multi)


# ----------------------------------------------------------------------
# Examples 자동 수집
# ----------------------------------------------------------------------
def _gather_examples(limit: int = 6) -> List[List[str]]:
    """각 클래스 폴더에서 첫 이미지 1장씩 모아 gr.Examples 형식으로 반환한다."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    root = EXAMPLES_ROOT if os.path.isdir(EXAMPLES_ROOT) else TEST_ROOT
    examples: List[List[str]] = []
    classes = CLASSIFIER.classes if CLASSIFIER is not None else []
    for english in classes:
        cls_dir = os.path.join(root, english)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(exts):
                examples.append([os.path.join(cls_dir, fname)])
                break
        if len(examples) >= limit:
            break
    return examples


# ----------------------------------------------------------------------
# UI 구성 (gr.Blocks)
# ----------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    supported_str = " · ".join(SUPPORTED_KOREAN) if SUPPORTED_KOREAN else "(모델 미로드)"
    examples = _gather_examples(limit=6)
    multi_note = "다중 음식 검출(YOLO11) 지원" if PIPELINE is not None else "단일 분류 모드"

    with gr.Blocks(title="한식 칼로리 분석기") as demo:
        gr.Markdown(
            "# 🍚 한식 칼로리 분석기 (Korean Food Calorie Analyzer)\n"
            "한식 사진을 업로드하면 **음식 종류와 칼로리**를 알려드립니다. "
            "여러 음식이 한 접시에 있으면 자동으로 **항목별 + 총 칼로리**를 계산합니다.\n\n"
            "- **모델:** YOLO11(검출) + EfficientNet-B0(분류) · 23종 한식 · "
            f"{multi_note}\n"
            f"- **지원 음식(23종):** {supported_str}"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="한식 사진 업로드", height=360)
                submit_btn = gr.Button("분석하기", variant="primary")
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=image_in,
                        label="예제 이미지 (클릭해서 바로 분석)",
                        examples_per_page=6,
                    )

            with gr.Column(scale=1):
                # --- 단일 경로 UI (기존 그대로) ---
                with gr.Group(visible=True) as single_group:
                    main_out = gr.Markdown(label="분석 결과")
                    topk_out = gr.Label(num_top_classes=3, label="Top-3 예측")
                    source_out = gr.Markdown(label="칼로리 출처")
                    with gr.Accordion("디버그 정보", open=False):
                        debug_out = gr.Markdown()

                # --- 다중 경로 UI (검출 ≥2개일 때) ---
                with gr.Group(visible=False) as multi_group:
                    multi_total = gr.Markdown(label="총 칼로리")
                    multi_image = gr.Image(label="검출 결과(박스)", height=320)
                    multi_table = gr.Dataframe(
                        headers=["#", "음식", "영문", "확률", "kcal", "YOLO참고"],
                        datatype=["number", "str", "str", "str", "str", "str"],
                        label="항목별 결과",
                        wrap=True,
                    )

        outputs = [
            single_group, main_out, topk_out, source_out, debug_out,
            multi_group, multi_image, multi_table, multi_total,
        ]
        submit_btn.click(fn=predict_food, inputs=image_in, outputs=outputs)
        image_in.change(fn=predict_food, inputs=image_in, outputs=outputs)

        gr.Markdown(
            "---\n"
            "**학습 데이터:** AI Hub 한국 음식 이미지 · "
            "**칼로리 출처:** 식약처(MFDS) 식품영양성분 DB + 일부 추정값 · "
            f"[GitHub]({GITHUB_URL})\n\n"
            "_칼로리는 표준 1인분 기준 참고값이며, 실제 조리·양에 따라 달라질 수 있습니다._"
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),     # Gradio 6.0: theme 은 launch() 로 이동
    )
