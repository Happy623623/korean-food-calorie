"""classifier 서브패키지: 음식 크롭 이미지를 15종 중 하나로 분류한다.

하이브리드 파이프라인의 2단계. YOLO 검출기가 잘라준 음식 크롭을
timm EfficientNet-B0 분류기로 15종 중 하나로 분류한다.

    dataset.py : 폴더 기반 분류 Dataset + albumentations 증강
    model.py   : EfficientNet-B0 분류기 (timm)
    train.py   : 학습 루프 (AdamW, CosineAnnealing, AMP, best.pt 저장)

클래스 인덱스 순서는 food_calories.json 의 항목 순서를 따른다
(= 칼로리 매핑 정렬 보장). 학습 데이터 폴더 이름은 각 음식의 영문명(english)을 쓴다.
"""
