# korean-food-calorie

한국 음식 사진에서 음식을 찾아 종류를 알아내고 칼로리를 추정하는 **하이브리드 딥러닝 파이프라인**.

```
이미지 ─▶ [1단계] YOLO11 검출 ─▶ 음식 영역 bbox
                                  │ 각 bbox 크롭
                                  ▼
            [2단계] EfficientNet-B0 분류 ─▶ 음식 종류(15종) + 확신도
                                  │ 검출 박스 1개 = 1인분
                                  ▼
                  kcal_per_serving 합산 ─▶ 음식별 kcal + 총 칼로리
```

## 왜 하이브리드인가

검출(localization)과 분류(classification)를 **한 모델에 모두 맡기지 않고 분리**했다.

- **YOLO11 (검출)** — 한 접시 위 여러 음식의 *위치*를 잡는 데 집중한다. 단일 클래스(`food`)만
  검출하므로 라벨링이 단순하고("음식이면 박스"), 종류 라벨 없이도 검출기를 학습할 수 있다.
- **EfficientNet-B0 (분류)** — 잘라낸 크롭 한 장을 15종 중 하나로 분류한다. 미세한 종류 구분이라는
  더 어려운 일에만 집중하므로, 폴더 기반 분류 데이터(`<class>/*.jpg`)로 쉽게 학습/교체할 수 있다.

각 단계를 독립적으로 학습·교체·평가할 수 있어 유지보수와 디버깅이 쉽다.
칼로리는 **검출된 박스 1개를 표준 1인분으로 보고** `kcal_per_serving` 을 합산한다(별도 양 회귀 없음).

> **단일 진실 공급원:** 클래스 인덱스(0~14)·이름·칼로리는 모두 `food_calories.json` 이 정한다.
> JSON 의 항목 순서 = 클래스 인덱스이고, 분류 데이터 폴더 이름은 각 항목의 영문명(`english`)을 쓴다.
> 분류기 체크포인트에 클래스 순서를 함께 저장해, 추론 시 정렬 불일치를 자동 검증한다.

## 음식 15종

김밥 · 떡볶이 · 비빔밥 · 김치찌개 · 된장찌개 · 불고기 · 제육볶음 · 삼겹살 · 냉면 · 칼국수 · 라면 · 순두부찌개 · 갈비탕 · 잡채 · 부대찌개

## 칼로리 매핑 데이터

본 프로젝트는 15종 한식의 1인분 평균 칼로리를 다음과 같이 매핑합니다.

### 데이터 출처

- **식품의약품안전처 식품영양성분 통합 DB** (https://various.foodsafetykorea.go.kr/nutrient/)
- 9개 항목은 식약처 공식 100g당 kcal × 1인분 무게로 계산 (`source: "MFDS"`)
- 6개 항목은 식약처 값에 다음 조정 사유가 있어 보정 (`source: "estimate"`)

### 보정 사유 (투명성)

| 음식 | 식약처 기준값 | 보정값 | 사유 |
|------|------------|--------|------|
| 갈비탕 | 198 kcal (33/100g × 600g) | 400 kcal | 식약처 값은 국물 위주 측정, 실제 1인분에 포함된 고기/국수 반영 |
| 삼겹살 | 934 kcal (467/100g × 200g) | 520 kcal | 식약처는 생고기 기준, 굽는 과정의 지방 감소 반영 |
| 라면 | 930 kcal (169/100g × 550g) | 500 kcal | 식약처 무게는 국물 포함, 일반 시판 라면 1봉지 기준 |
| 순두부찌개 | 100 kcal (25/100g × 400g) | 250 kcal | 식약처 값이 일반 식당 1인분 대비 낮음 |
| 칼국수 | 679 kcal (97/100g × 700g) | 480 kcal | 일반 참고값 대비 보수적 조정 |
| 불고기 | - | 430 kcal | 식약처에 "돼지불고기"만 등록, 일반적 의미의 소불고기 추정값 사용 |

각 항목은 `food_calories.json`의 `source`, `note` 필드에서 확인 가능합니다.

### 한계

- 1인분 평균 기준의 고정값 매핑이며, 실제 사진의 음식 양에 따른 정밀한 portion 추정은 수행하지 않습니다.
- 향후 개선 사항: depth estimation 또는 segmentation 기반 부피 추정으로 정확도 향상 가능.

## 설치

```bash
pip install -r requirements.txt
```

## 디렉터리 구조

```
korean-food-calorie/
├── infer.py                     # 추론 CLI 진입점
├── food_calories.json           # 클래스/칼로리 단일 진실 공급원 (15종)
├── configs/
│   ├── detector.yaml            # 1단계 YOLO 검출기 설정
│   └── classifier.yaml          # 2단계 분류기 설정
├── src/
│   ├── detector/                # 1단계: 음식 영역 검출 (YOLO11)
│   │   ├── train_yolo.py        #   학습 + dataset.yaml 자동 생성
│   │   └── inference_yolo.py    #   박스 추론 (FoodDetector)
│   ├── classifier/              # 2단계: 음식 15종 분류 (EfficientNet-B0)
│   │   ├── dataset.py           #   폴더 기반 Dataset + albumentations
│   │   ├── model.py             #   timm EfficientNet-B0 분류기
│   │   └── train.py             #   AdamW · CosineAnnealing · AMP · best.pt
│   ├── pipeline.py              # 검출→crop→분류→kcal 합산 (FoodCaloriePipeline)
│   └── utils.py                 # 시드/설정/로깅/체크포인트 + FoodTable
└── data/
    ├── detector/                # YOLO 검출 학습 데이터
    │   ├── images/{train,val,test}/
    │   └── labels/{train,val,test}/
    └── classifier/              # 폴더 기반 분류 학습 데이터
        ├── train/<english_name>/*.jpg
        └── val/<english_name>/*.jpg
```

## 데이터 준비

### 1단계 — 검출기 (YOLO 형식, 단일 클래스 `food`)

이미지와 라벨을 같은 파일명으로 1:1 배치한다.

```
data/detector/
  images/{train,val,test}/  *.jpg
  labels/{train,val,test}/  *.txt
```

라벨(`.txt`) 한 줄 — 모두 0~1 정규화, 단일 클래스이므로 `class_id` 는 항상 `0`:

```
0 <center_x> <center_y> <width> <height>
```

### 2단계 — 분류기 (폴더 기반)

음식 종류별 폴더(영문명)에 크롭 이미지를 담는다. 폴더 이름은 `food_calories.json` 의 `english` 와 일치시킨다.

```
data/classifier/
  train/
    gimbap/ *.jpg
    tteokbokki/ *.jpg
    bibimbap/ *.jpg
    ...
  val/
    gimbap/ *.jpg
    ...
```

> 분류용 크롭은 검출기 학습 라벨에서 bbox 를 잘라 만들거나, AI Hub 등 한국 음식 데이터셋의
> 종류별 이미지를 위 폴더 구조로 배치해 확보한다.

## 학습

```bash
# 1단계: 음식 영역 검출기 (configs/classes 로부터 dataset.yaml 자동 생성)
python -m src.detector.train_yolo --config configs/detector.yaml

# 2단계: 음식 15종 분류기 (AdamW + CosineAnnealing + AMP, best.pt 저장)
python -m src.classifier.train --config configs/classifier.yaml
```

- 검출기: COCO 사전학습 `yolo11s.pt` 에서 fine-tuning, 완료 후 best.pt 를
  `configs/detector.yaml` 의 `model.weights`(기본 `checkpoints/yolo11_food.pt`)로 복사.
- 분류기: ImageNet 사전학습 EfficientNet-B0 에서 fine-tuning, 검증 top-1 정확도 최고 시점을
  `configs/classifier.yaml` 의 `train.weights`(기본 `checkpoints/classifier_best.pt`)로 저장.

## 추론

```bash
# 결과(JSON) 출력
python infer.py --source path/to/food.jpg

# 박스가 그려진 이미지로 저장
python infer.py --source path/to/food.jpg --save outputs/annotated.jpg

# 폴더 일괄 처리
python infer.py --source path/to/folder/
```

출력 예:

```json
{
  "detections": [
    {
      "class_id": 2,
      "name": "비빔밥",
      "english": "bibimbap",
      "detect_conf": 0.91,
      "class_prob": 0.97,
      "bbox_xyxy": [34.0, 50.5, 410.2, 380.0],
      "kcal": 600.0,
      "grams": 450.0
    }
  ],
  "total_kcal": 600.0,
  "num_items": 1,
  "image_shape": [480, 640]
}
```

- `detect_conf` : 1단계 YOLO 의 음식 검출 신뢰도
- `class_prob`  : 2단계 분류기의 종류 확신도(softmax)
- `kcal`        : 해당 음식 1인분 평균 칼로리 (`food_calories.json`)

## 기술 스택

PyTorch · Ultralytics YOLO11 · timm (EfficientNet-B0) · albumentations · OpenCV
