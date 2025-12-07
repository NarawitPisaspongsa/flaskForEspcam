from ultralytics import YOLO
import base64
import tempfile
import os

# Load model once at cold start (serverless best practice)
model = YOLO(os.path.join(os.path.dirname(__file__), "..", "best.pt"))
model.to("cpu")
model.overrides["verbose"] = False
model.overrides["imgsz"] = 640
model.overrides["half"] = False
model.overrides["device"] = "cpu"
model.overrides["augment"] = False
model.overrides["visualize"] = False


def handler(request, response):
    if request.method != "POST":
        return response.status(405).json({"error": "POST only"})

    try:
        body = request.get_json()
        if "image_base64" not in body:
            return response.status(400).json({"error": "missing image_base64"})

        # Decode image
        img_data = base64.b64decode(body["image_base64"])

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(img_data)
            img_path = tmp.name

        # YOLO inference
        results = model.predict(
            source=img_path, imgsz=640, conf=0.25, verbose=False, device="cpu"
        )

        preds = results[0]
        top_class = preds.names[preds.probs.top1]

        return response.status(200).json({"predict_class": top_class})

    except Exception as e:
        return response.status(500).json({"error": str(e)})
