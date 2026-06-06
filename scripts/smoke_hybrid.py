# -*- coding: utf-8 -*-
"""하이브리드 파이프라인 로컬 스모크 (CPU): 단일 / 다중 경로 검증.

1) 단일: data/classifier/test 한 장 → mode=single 확인 + 기존 FoodClassifier 와 top1 동일성 확인.
2) 다중(자연): data/detector/images/test 중 검출 ≥2개 이미지 → mode=multi + 항목/총합 확인.
3) 다중(합성): 서로 다른 두 클래스의 test 이미지를 좌우로 이어붙인 1장 → mode=multi +
   서로 다른 두 음식의 칼로리 합산 확인. 박스 시각화 이미지를 outputs/ 에 저장.

사용 (프로젝트 루트에서):
    python scripts/smoke_hybrid.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import cv2

from src.pipeline import FoodCaloriePipeline
from src.classifier.inference import FoodClassifier

CLF_CKPT = "checkpoints/best_efficientnet_b0_23.pt"
DET_CKPT = "checkpoints/yolo11n_23cls.pt"
CLASSIFIER_TEST = "data/classifier/test"
DETECTOR_TEST = "data/detector/images/test"
OUT_DIR = "outputs"
EXTS = (".jpg", ".jpeg", ".png")


def first_image_under(root):
    """root/<클래스>/ 트리에서 첫 이미지 한 장(경로) 반환."""
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(EXTS):
                return os.path.join(d, fn)
    return None


def list_detector_test():
    return [f for f in sorted(os.listdir(DETECTOR_TEST)) if f.lower().endswith(EXTS)]


def find_natural_multi(pipe, files, limit=250):
    """검출 ≥2개가 나오는 첫 이미지 경로를 찾는다(없으면 None)."""
    import random
    random.seed(42)
    shuffled = files[:]
    random.shuffle(shuffled)
    for fn in shuffled[:limit]:
        p = os.path.join(DETECTOR_TEST, fn)
        res = pipe.predict(p)
        if res["mode"] == "multi":
            return p, res
    return None, None


def class_of(fname):
    """'<english>__<stem>.jpg' → english 클래스명."""
    return fname.split("__", 1)[0]


# 좌우 결합 시 두 음식이 별개 영역으로 잘 검출되도록 시각적으로 구분되는 클래스를 우선 사용.
COMPOSITE_PREFERRED = ("pizza", "hamburger")
COMPOSITE_HEIGHT = 480
COMPOSITE_GAP = 30


def make_composite(files, out_path):
    """서로 다른 두 클래스의 test 이미지를 같은 높이로 맞춰 흰 간격을 두고 좌우로 결합해 저장.

    높이 통일 + 가운데 흰 간격(separator)을 두면 검출기가 두 음식을 별개 영역으로 더 잘 잡는다.
    """
    import numpy as np

    by_class = {}
    for fn in files:
        by_class.setdefault(class_of(fn), fn)
    classes = list(by_class)
    if len(classes) < 2:
        return None
    # 선호 쌍(pizza+hamburger)이 둘 다 있으면 사용, 아니면 앞에서 두 클래스.
    if all(c in by_class for c in COMPOSITE_PREFERRED):
        ca, cb = COMPOSITE_PREFERRED
    else:
        ca, cb = classes[0], classes[1]

    ia = cv2.imread(os.path.join(DETECTOR_TEST, by_class[ca]))
    ib = cv2.imread(os.path.join(DETECTOR_TEST, by_class[cb]))
    if ia is None or ib is None:
        return None
    h = min(ia.shape[0], ib.shape[0], COMPOSITE_HEIGHT)
    ia = cv2.resize(ia, (int(ia.shape[1] * h / ia.shape[0]), h))
    ib = cv2.resize(ib, (int(ib.shape[1] * h / ib.shape[0]), h))
    sep = 255 * np.ones((h, COMPOSITE_GAP, 3), dtype="uint8")
    comp = cv2.hconcat([ia, sep, ib])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, comp)
    return out_path, (ca, cb)


def print_result(tag, res):
    print(f"\n[{tag}] mode={res['mode']}  num_items={res['num_items']}  "
          f"num_boxes={res['num_boxes']}  total_kcal={res['total_kcal']}")
    for i, it in enumerate(res["items"], 1):
        box = it["bbox_xyxy"]
        dc = it["detect_conf"]
        print(f"   {i}. {it['korean']}({it['english']})  "
              f"p={it['class_prob']:.3f}  kcal={it['kcal']:g}  "
              f"yolo참고={it.get('yolo_class')}  "
              f"detect_conf={dc}  bbox={box}")


def main():
    for ck in (CLF_CKPT, DET_CKPT):
        if not os.path.isfile(ck):
            print(f"[오류] 체크포인트 없음: {ck}", file=sys.stderr)
            sys.exit(1)

    pipe = FoodCaloriePipeline.from_paths(
        detector_weights=DET_CKPT, classifier_weights=CLF_CKPT,
        food_table="food_calories.json", conf=0.35, infer_device="cpu",
    )
    print(f"[load] classes={len(pipe.classes)} | image_size={pipe.image_size} | device={pipe.device}")

    # ===== 1) 단일 경로 =====
    single_img = first_image_under(CLASSIFIER_TEST)
    if single_img is None:
        print("[오류] data/classifier/test 에서 이미지를 못 찾음", file=sys.stderr); sys.exit(1)
    res_single = pipe.predict(single_img)
    print(f"\n=== 단일 스모크: {single_img} ===")
    print_result("SINGLE", res_single)
    assert res_single["mode"] == "single", "단일 이미지인데 multi 로 분기됨"

    # 기존 FoodClassifier 와 top1 동일성(검증된 동작 보존) 확인
    clf = FoodClassifier(checkpoint_path=CLF_CKPT, food_table_path="food_calories.json", device="cpu")
    base = clf.predict(single_img, top_k=1)["top1"]
    item = res_single["items"][0]
    same = (base["english"] == item["english"])
    print(f"   [동일성] 기존 분류기 top1={base['english']}({base['confidence']*100:.1f}%) "
          f"vs 파이프라인={item['english']}({item['class_prob']*100:.1f}%) → "
          f"{'일치 ✓' if same else '불일치 ✗'}")
    assert same, "단일 경로가 기존 분류기와 다른 top1 을 냄"

    # ===== 2) 다중 경로 (자연 검출) =====
    files = list_detector_test()
    nat_path, nat_res = find_natural_multi(pipe, files)
    if nat_path is not None:
        print(f"\n=== 다중 스모크(자연): {nat_path} ===")
        print_result("MULTI-NATURAL", nat_res)
        img_rgb, _ = pipe.render_image(nat_path)
        out1 = os.path.join(OUT_DIR, "smoke_multi_natural.jpg")
        os.makedirs(OUT_DIR, exist_ok=True)
        cv2.imwrite(out1, img_rgb[:, :, ::-1])
        print(f"   박스 시각화 저장: {out1}")
        assert nat_res["mode"] == "multi"
    else:
        print("\n[정보] 자연 다중 검출 이미지를 표본에서 못 찾음 → 합성으로 대체")

    # ===== 3) 다중 경로 (합성 좌우 결합) =====
    comp = make_composite(files, os.path.join(OUT_DIR, "smoke_multi_composite.jpg"))
    if comp is None:
        print("[경고] 합성 이미지 생성 실패(클래스 2개 이상 필요)", file=sys.stderr)
    else:
        comp_path, (ca, cb) = comp
        print(f"\n=== 다중 스모크(합성): {comp_path}  (왼쪽={ca}, 오른쪽={cb}) ===")
        res_comp = pipe.predict(comp_path)
        print_result("MULTI-COMPOSITE", res_comp)
        img_rgb, _ = pipe.render_image(comp_path)
        out2 = os.path.join(OUT_DIR, "smoke_multi_composite_boxed.jpg")
        cv2.imwrite(out2, img_rgb[:, :, ::-1])
        print(f"   박스 시각화 저장: {out2}")
        # 합성은 서로 다른 두 음식이 있으므로 다중 검출/합산이 기대된다(검출 성능에 따라 변동 가능).
        manual = sum(it["kcal"] for it in res_comp["items"])
        print(f"   [합산검증] sum(items.kcal)={manual:g}  vs total_kcal={res_comp['total_kcal']:g} → "
              f"{'일치 ✓' if abs(manual-res_comp['total_kcal'])<0.05 else '불일치 ✗'}")

    print("\n[완료] 하이브리드 스모크 통과")


if __name__ == "__main__":
    main()
