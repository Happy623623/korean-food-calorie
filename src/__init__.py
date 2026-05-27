"""korean-food-calorie 패키지.

한국 음식 15종에 대한 하이브리드 칼로리 추정 파이프라인을 제공한다.

    1단계 (YOLO11)         : 음식 영역 검출(localize) -> src/detector/
    2단계 (EfficientNet-B0): 크롭 이미지를 15종으로 분류 -> src/classifier/
    통합                   : 검출 -> crop -> 분류 -> 1인분 평균 kcal 합산 -> src/pipeline.py

portion 회귀는 하지 않는다. 검출된 박스 1개 = 1인분으로 kcal 을 합산한다.
"""

__version__ = "0.3.0"
