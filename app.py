from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile

app = Flask(__name__)

# Load model once when server starts
model = YOLO("best.pt")


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="no image"), 400

    image = request.files["image"]

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        path = tmp.name

    results = model(path)
    preds = results[0]
    top_class = preds.names[preds.probs.top1]

    return jsonify(predicted_class=top_class)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
