from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile
import torch

app = Flask(__name__)

# Load model in lightweight, CPU-friendly mode
model = YOLO("best.pt")
model.to("cpu")
model.overrides["verbose"] = False
model.overrides["imgsz"] = 640  # force smaller inference size
model.overrides["half"] = False  # CPU cannot do half precision
model.overrides["device"] = "cpu"
model.overrides["augment"] = False
model.overrides["visualize"] = False


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="no image"), 400

    image = request.files["image"]

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    # CPU-safe inference
    results = model.predict(
        source=img_path, imgsz=640, conf=0.25, verbose=False, device="cpu"
    )

    preds = results[0]
    top_class = preds.names[preds.probs.top1]

    return jsonify(predicted_class=top_class)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
