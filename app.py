"""한식 칼로리 분석기 Gradio 데모 (Phase 4-2).

src.classifier.inference.FoodClassifier 를 사용해 한식 사진 한 장을 받아
음식 종류 + 1인분 칼로리 + Top-3 예측을 웹 UI 로 보여준다.

실행 (프로젝트 루트에서):
    python app.py
    → http://127.0.0.1:7860 접속

분류기는 앱 시작 시 1번만 로드해 전역으로 재사용한다.
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
CHECKPOINT_PATH = "checkpoints/best_efficientnet_b0_23.pt"
FOOD_TABLE_PATH = "food_calories.json"
TEST_ROOT = "data/classifier/test"          # gr.Examples 자동 수집 대상
LOW_CONFIDENCE_THRESHOLD = 0.50             # 이 미만이면 경고 표시
GITHUB_URL = "https://github.com/your-username/korean-food-calorie"  # TODO: 실제 저장소로 교체

# 칼로리 출처 코드 → 사람이 읽기 좋은 한글 라벨
SOURCE_LABELS = {
    "MFDS": "식약처(MFDS) 공식 데이터",
    "estimate": "추정값(estimate)",
}

# ----------------------------------------------------------------------
# 모델 로드 (전역 1회)
# ----------------------------------------------------------------------
try:
    CLASSIFIER: Optional[FoodClassifier] = FoodClassifier(
        checkpoint_path=CHECKPOINT_PATH,
        food_table_path=FOOD_TABLE_PATH,
        device="cpu",                       # CPU 추론 기준
    )
    SUPPORTED_KOREAN: List[str] = [
        CLASSIFIER.food_table.get(c, {}).get("korean") or c
        for c in CLASSIFIER.classes
    ]
except Exception as exc:  # 체크포인트/테이블 누락 등 → UI 는 띄우되 추론 시 안내
    logger.error("FoodClassifier 로드 실패: %s", exc)
    CLASSIFIER = None
    SUPPORTED_KOREAN = []


# ----------------------------------------------------------------------
# 출처 표기 헬퍼
# ----------------------------------------------------------------------
def _source_label(source: Optional[str]) -> str:
    """출처 코드를 한글 라벨로 변환한다(미지정/미상은 그대로)."""
    if not source:
        return "출처 미상"
    return SOURCE_LABELS.get(source, source)


# ----------------------------------------------------------------------
# 추론 함수 (Gradio 콜백)
# ----------------------------------------------------------------------
def predict_food(image) -> Tuple[str, Dict[str, float], str, str]:
    """업로드된 PIL 이미지를 분류해 4개 출력을 반환한다.

    Returns:
        (메인 마크다운, Top-3 라벨 dict, 칼로리 출처 마크다운, 영어 클래스명)
    """
    # 1) 안전장치: 모델 미로드
    if CLASSIFIER is None:
        msg = (
            "### ⚠️ 모델을 불러오지 못했습니다\n"
            f"`{CHECKPOINT_PATH}` 체크포인트와 `{FOOD_TABLE_PATH}` 를 확인하세요."
        )
        return msg, {}, "", ""

    # 2) 안전장치: 이미지 미업로드
    if image is None:
        return "### 👆 한식 사진을 업로드하면 분석 결과가 여기에 표시됩니다.", {}, "", ""

    # 3) 추론
    try:
        result = CLASSIFIER.predict(image, top_k=3)
    except Exception as exc:  # 예기치 못한 추론 오류
        logger.exception("추론 중 오류")
        msg = (
            "### ❌ 분석 중 오류가 발생했습니다\n"
            f"```\n{exc}\n```\n다른 이미지로 다시 시도해 주세요."
        )
        return msg, {}, "", ""

    top1 = result["top1"]
    korean = top1["korean"] or "알 수 없음"
    english = top1["english"]
    confidence = top1["confidence"]

    # 4) 메인 결과 마크다운
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

    # 낮은 신뢰도 경고
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        lines.append("")
        lines.append(
            "> ⚠️ 모델 신뢰도가 낮습니다. 학습된 23종 한식이 아닐 수 있습니다."
        )

    main_md = "\n".join(lines)

    # 5) Top-3 라벨 dict (gr.Label 용): {한글명: 확률}
    top_k_labels: Dict[str, float] = {}
    for entry in result["top_k"]:
        label = entry["korean"] or entry["english"]
        top_k_labels[label] = float(entry["confidence"])

    # 6) 칼로리 출처 표시
    if kcal is not None:
        source_md = (
            f"**칼로리 출처:** {_source_label(top1['source'])}\n\n"
            "식약처 데이터는 공식 영양성분 DB 기준이며, "
            "`추정값`은 조리 방식·1인분량을 반영해 보정한 값입니다."
        )
    else:
        source_md = "**칼로리 출처:** 정보 없음"

    # 7) 디버깅용 영어 클래스명
    debug_md = f"`english_class = {english}`"

    return main_md, top_k_labels, source_md, debug_md


# ----------------------------------------------------------------------
# Examples 자동 수집: data/classifier/test/{english}/ 첫 파일 1장씩
# ----------------------------------------------------------------------
def _gather_examples(limit: int = 6) -> List[List[str]]:
    """각 클래스 test 폴더에서 첫 이미지 1장씩 모아 gr.Examples 형식으로 반환한다.

    Gradio Examples 는 [[입력1], [입력2], ...] 형태(입력 컴포넌트 1개 → 1-원소 리스트).
    클래스 폴더가 없거나 비어 있으면 조용히 건너뛴다.
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    examples: List[List[str]] = []
    classes = CLASSIFIER.classes if CLASSIFIER is not None else []
    for english in classes:
        cls_dir = os.path.join(TEST_ROOT, english)
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

    with gr.Blocks(title="한식 칼로리 분석기", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🍚 한식 칼로리 분석기 (Korean Food Calorie Analyzer)\n"
            "한식 사진을 업로드하면 **음식 종류와 1인분 칼로리**를 알려드립니다.\n\n"
            "- **모델:** EfficientNet-B0 분류기 (torchvision) · 23종 한식\n"
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
                main_out = gr.Markdown(label="분석 결과")
                topk_out = gr.Label(num_top_classes=3, label="Top-3 예측")
                source_out = gr.Markdown(label="칼로리 출처")
                with gr.Accordion("디버그 정보", open=False):
                    debug_out = gr.Markdown()

        # 버튼 클릭 + 이미지 변경(예제 클릭 포함) 모두에서 추론 실행
        submit_btn.click(
            fn=predict_food,
            inputs=image_in,
            outputs=[main_out, topk_out, source_out, debug_out],
        )
        image_in.change(
            fn=predict_food,
            inputs=image_in,
            outputs=[main_out, topk_out, source_out, debug_out],
        )

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
    # share=False: 우선 로컬에서 테스트.
    # 같은 네트워크의 다른 기기에서 접근하려면 server_name="0.0.0.0" 로 띄운다.
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
