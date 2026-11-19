"""Draw side-by-side annotations and predictions on four shared test photos."""
from PIL import Image, ImageDraw, ImageFont

from comparer_confusions import predictions
from auditer_annotations import read_labels
from comparer_robin import OUT, ROBIN, ROOT, names_for

COLORS = {"banana": "#ba7611", "orange": "#bf315b", "red_apple": "#1367a3",
          "green_apple": "#138369"}


def main():
    cases = ["image123.jpg", "image209.jpg", "image3.jpg", "image51.jpg"]
    ours, robin = ROOT/"dataset", ROBIN/"dataset"
    output = ROOT/"predictions"
    ours_preds = predictions(output/"YOLO_Predictions/detections.csv")
    robin_preds = predictions(ROBIN/"predictions/Robin_YOLO26/detections.csv")
    canvas = Image.new("RGB", (1040, 620), "white")
    pen = ImageDraw.Draw(canvas)
    header = ImageFont.truetype("DejaVuSans.ttf", 18)
    small = ImageFont.truetype("DejaVuSans.ttf", 16)
    columns = ["Annotations Alexis", "Annotations Robin", "Prédictions Alexis", "Prédictions Robin"]
    for i, label in enumerate(columns):
        pen.text((112 + i*231, 7), label, fill="#182334", font=header)
    for i, name in enumerate(cases):
        with Image.open(ours/"images/test"/name) as source:
            base = source.convert("RGB")
        base.thumbnail((217, 134))
        y = 32 + 146*i
        pen.text((7, y+5), name.replace(".jpg", ""), fill="#182334", font=small)
        series = [read_labels(ds, "test", name, names_for(ds))[0] for ds in (ours, robin)]
        series += [ours_preds[name], robin_preds[name]]
        for j, boxes in enumerate(series):
            image = base.copy()
            draw = ImageDraw.Draw(image)
            w, h = image.size
            for cls, (x1, y1, x2, y2) in boxes:
                draw.rectangle((round(x1*w), round(y1*h), round(x2*w), round(y2*h)),
                               outline=COLORS[cls], width=3)
            canvas.paste(image, (111 + j*231 + (217-w)//2, y))
    canvas.save(OUT/"figures/erreurs_communes.jpg", quality=95)


if __name__ == "__main__":
    main()
