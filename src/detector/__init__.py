"""detector 서브패키지: 이미지에서 '음식 영역'을 검출(localize)한다.

하이브리드 파이프라인의 1단계. Ultralytics YOLO11 을 단일 클래스('food')
검출기로 학습/사용한다. 음식 "종류" 분류는 하지 않고 위치(bbox)만 찾으며,
종류 분류는 classifier 서브패키지가 크롭 이미지로 담당한다.

    train_yolo.py     : YOLO11 검출기 학습 + ultralytics dataset.yaml 자동 생성
    inference_yolo.py : 학습된 검출기로 박스 추론 (FoodDetector)

detector.yaml 의 `classes` 를 여러 개로 늘리면 다중 클래스 검출기로도 학습할 수 있으나,
하이브리드 설계에서는 classifier 가 분류를 맡으므로 기본값은 단일 클래스다.
"""
