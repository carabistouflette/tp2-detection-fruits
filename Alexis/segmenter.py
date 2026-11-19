"""Reproduce the YOLO + SAM 2.1 Tiny example and the manual-mask evaluation."""
import argparse
import contextlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as rle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.models.sam.predict import SAM2Predictor

from evaluate import ROOT, selected_weights

CLASSES = ("red_apple", "green_apple", "banana", "orange")
COLORS = ((60, 70, 255), (80, 200, 30), (30, 235, 240), (30, 140, 255))


def image_at_scale(path):
    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    ratio = min(1, 768 / max(h, w))
    if ratio < 1:
        image = cv2.resize(image, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_AREA)
    return image


def encode(mask):
    return rle.encode(np.asfortranarray(mask, dtype=np.uint8))


def prepare_coco(dataset, split):
    coco = COCO()
    coco.dataset = {"images": [], "annotations": [], "info": {},
                    "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES, 1)]}
    photos = []
    for i, photo in enumerate(sorted((dataset / "images" / split).glob("*.jpg")), 1):
        image = image_at_scale(photo)
        h, w = image.shape[:2]
        photos.append((i, photo.name, image))
        coco.dataset["images"].append({"id": i, "width": w, "height": h, "file_name": photo.name})
        label = (dataset / "labels" / split / f"{photo.stem}.txt").read_text().splitlines()
        for line in label:
            cls, *coords = map(float, line.split())
            mask = np.zeros((h, w), np.uint8)
            polygon = np.rint(np.array(coords).reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.fillPoly(mask, [polygon], 1)
            encoded = encode(mask)
            coco.dataset["annotations"].append({
                "id": len(coco.dataset["annotations"]) + 1, "image_id": i, "category_id": int(cls) + 1,
                "segmentation": encoded, "area": float(rle.area(encoded)), "bbox": rle.toBbox(encoded).tolist(),
                "iscrowd": 0,
            })
    coco.createIndex()
    return coco, photos


class MultiMaskPredictor(SAM2Predictor):
    def inference(self, im, *args, **kwargs):
        kwargs["multimask_output"] = True
        return super().inference(im, *args, **kwargs)


def mask_iou(first, second):
    intersection = np.count_nonzero(first & second)
    union = np.count_nonzero(first | second)
    return intersection / max(1, union)


def contour_options(masks, identities, confidence, boxes):
    options = [[] for _ in boxes]
    for mask, identity, conf in zip(masks, identities, confidence):
        i = int(identity) // 3
        assert 0 <= i < len(boxes)
        left, top, right, bottom = boxes[i]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 6:
            continue
        polygon = cv2.approxPolyDP(contour, max(0.45, 0.0018 * cv2.arcLength(contour, True)), True)
        if len(polygon) < 3:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        intersection = max(0, min(x + w, right) - max(x, left)) * max(0, min(y + h, bottom) - max(y, top))
        iou = intersection / max(1, w * h + (right - left) * (bottom - top) - intersection)
        shape = np.zeros_like(mask, dtype=np.uint8)
        cv2.fillPoly(shape, [polygon], 1)
        score = 0.55 * float(conf) + 0.45 * iou
        options[i].append((score, float(conf), iou, polygon[:, 0, :], shape.astype(bool)))
    return [sorted(group, key=lambda x: x[0], reverse=True) for group in options]


def detection_boxes(detector, image, device, size):
    output = detector.predict(image, imgsz=size, conf=0.025, max_det=40, device=device, verbose=False)[0].boxes
    return output.xyxy.cpu().numpy(), output.cls.cpu().numpy(), output.conf.cpu().numpy()


def two_stage(detector, sam, image, device, detector_size=640):
    boxes, labels, scores = detection_boxes(detector, image, device, detector_size)
    if not len(boxes):
        return []
    segmented = sam(source=image, bboxes=boxes, verbose=False)[0]
    if segmented.masks is None:
        return []
    choices = contour_options(segmented.masks.data.cpu().numpy(), segmented.boxes.cls.cpu().numpy(),
                              segmented.boxes.conf.cpu().numpy(), boxes)
    masks = []
    for i in np.argsort(-scores):
        for _, conf, iou, _, mask in choices[i]:
            if iou >= 0.38 and all(mask_iou(mask, m) < 0.84 for c, _, m in masks if c == labels[i]):
                masks.append((labels[i], float(scores[i]) * np.sqrt(max(conf, 0)), mask))
                break
    return masks


def metric(coco, photos, function):
    predictions = []
    for index, name, image in photos:
        for cls, score, mask in function(name, image):
            compressed = encode(mask)
            predictions.append({"image_id": index, "category_id": int(cls) + 1,
                                "score": float(score), "segmentation": compressed})
        if index % 15 == 0:
            print(f"{index}/{len(photos)} images", flush=True)
    if not predictions:
        return {"map50_95": 0.0, "map50": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        evaluation = COCOeval(coco, coco.loadRes(predictions), "segm")
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    return {"map50_95": float(evaluation.stats[0]), "map50": float(evaluation.stats[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("--evaluer", action="store_true")
    parser.add_argument("--poids-yolo", type=Path, default=selected_weights())
    parser.add_argument("--sam", type=Path, default=ROOT / "sam2.1_t.pt")
    parser.add_argument("--taille-yolo", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sortie", type=Path, default=Path("segmentation.jpg"))
    args = parser.parse_args()
    if bool(args.image) == args.evaluer:
        parser.error("Indiquer une image ou utiliser --evaluer.")
    torch.set_num_threads(6)
    torch.backends.nnpack.set_flags(False)
    detector = YOLO(str(args.poids_yolo))
    if tuple(detector.names.values()) != CLASSES:
        parser.error("L'ordre des classes du détecteur ne correspond pas aux données d'Alexis.")
    sam = MultiMaskPredictor(overrides={"model": str(args.sam), "device": args.device,
                                       "imgsz": 768, "task": "segment", "mode": "predict",
                                       "conf": 0.01, "save": False, "verbose": False})

    def predict(image):
        return two_stage(detector, sam, image, args.device, args.taille_yolo)

    if args.evaluer:
        truth, photos = prepare_coco(ROOT / "reference_segmentation", "test")
        values = metric(truth, photos, lambda _, image: predict(image))
        print(json.dumps(values, indent=2))
        return
    image = image_at_scale(args.image)
    drawn = image.copy()
    for i, (_, score, mask) in enumerate(predict(image)):
        if score < 0.25:
            continue
        color = COLORS[i % len(COLORS)]
        drawn[mask] = (0.82 * drawn[mask] + 0.18 * np.asarray(color)).astype(np.uint8)
        lines, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(drawn, lines, -1, color, 2, cv2.LINE_AA)
    if not cv2.imwrite(str(args.sortie), drawn):
        raise OSError(args.sortie)


if __name__ == "__main__":
    main()
