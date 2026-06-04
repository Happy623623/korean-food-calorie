# -*- coding: utf-8 -*-
"""23종 분류기 로컬 스모크 테스트 (CPU).

checkpoints/best_efficientnet_b0_23.pt 로 신규 클래스 포함 test 이미지 몇 장을 예측해
[예측 클래스 + 확률(Top-3) + 1인분 칼로리] 가 정상 출력되는지 확인한다.

사용 (프로젝트 루트에서):
    python scripts/smoke_test.py                       # pizza/donkkaseu/rice 자동 1장씩
    python scripts/smoke_test.py img1.jpg img2.jpg ... # 지정 이미지들
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from src.classifier.inference import FoodClassifier

CKPT = "checkpoints/best_efficientnet_b0_23.pt"
TEST_ROOT = "data/classifier/test"
DEFAULT_CLASSES = ("pizza", "donkkaseu", "rice")  # 신규 클래스 위주


def first_image(cls: str):
    d = os.path.join(TEST_ROOT, cls)
    if not os.path.isdir(d):
        return None
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(d, fn)
    return None


def main():
    if not os.path.isfile(CKPT):
        print(f"[오류] 체크포인트가 없습니다: {CKPT}\n"
              f"       Colab 에서 학습한 .pt 를 이 경로에 두고 다시 실행하세요.", file=sys.stderr)
        sys.exit(1)

    images = sys.argv[1:]
    if not images:
        images = [p for c in DEFAULT_CLASSES if (p := first_image(c))]
    if not images:
        print("[오류] 테스트할 이미지를 찾지 못했습니다.", file=sys.stderr)
        sys.exit(1)

    clf = FoodClassifier(checkpoint_path=CKPT, food_table_path="food_calories.json", device="cpu")
    print(f"[load] classes={len(clf.classes)} | image_size={clf.image_size} | device={clf.device}\n")

    for img in images:
        res = clf.predict(img, top_k=3)
        t1 = res["top1"]
        kcal = t1["kcal_per_serving"]
        kcal_str = f"{kcal:g} kcal" if kcal is not None else "N/A"
        print("=" * 60)
        print(f"이미지 : {img}")
        print(f"예측   : {t1['korean']}({t1['english']})  확률 {t1['confidence']*100:.2f}%")
        print(f"칼로리 : 1인분 {kcal_str}"
              + (f"  ({t1['serving_size_g']:g}g, 출처 {t1['source']})" if kcal is not None else ""))
        print("Top-3  :")
        for r in res["top_k"]:
            kc = r["kcal_per_serving"]
            print(f"   {r['korean'] or '?'}({r['english']})  {r['confidence']*100:5.2f}%  "
                  f"| {kc:g} kcal" if kc is not None else
                  f"   {r['english']}  {r['confidence']*100:5.2f}%  | N/A")
    print("=" * 60)


if __name__ == "__main__":
    main()
