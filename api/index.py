from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)


@app.route("/")
def home():
    return "Flask Vercel Example - Image Upload API", 200


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # run your model here
        # prediction = model.predict(img)

        # Example dummy output
        return jsonify({
            "status": "success",
            "message": "Image received successfully",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"status": 404, "message": "Not Found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
