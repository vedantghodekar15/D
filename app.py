from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "baseline_ann.keras"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
model = None
model_error = None


def load_model():
    global model, model_error
    if model is not None or model_error is not None:
        return model
    if not MODEL_PATH.exists():
        model_error = f"Model not found at {MODEL_PATH}. Run: python train_model.py"
        return None
    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as error:
        model_error = f"Unable to load model: {error}"
    return model


def preprocess_image(image_file):
    image = Image.open(image_file).convert("L")
    pixels = np.asarray(image, dtype="uint8")

    # MNIST uses bright digits on a dark background. Adapt common scanned
    # images that use dark ink on a light background before resizing.
    corner_pixels = np.concatenate([
        pixels[:10, :10].ravel(),
        pixels[:10, -10:].ravel(),
        pixels[-10:, :10].ravel(),
        pixels[-10:, -10:].ravel(),
    ])
    if corner_pixels.mean() > 127 and pixels.mean() < corner_pixels.mean():
        pixels = 255 - pixels

    foreground = pixels > 30
    rows, columns = np.where(foreground)
    if rows.size:
        cropped = Image.fromarray(pixels).crop((
            columns.min(), rows.min(), columns.max() + 1, rows.max() + 1
        ))
        scale = min(20 / cropped.width, 20 / cropped.height)
        resized = cropped.resize((max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))))
        canvas = Image.new("L", (28, 28), color=0)
        position = ((28 - resized.width) // 2, (28 - resized.height) // 2)
        canvas.paste(resized, position)
        pixels = np.asarray(canvas, dtype="uint8")
    else:
        pixels = np.asarray(ImageOps.fit(Image.fromarray(pixels), (28, 28)), dtype="uint8")

    return (pixels.astype("float32") / 255.0).reshape(1, 784)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Image must be smaller than 5 MB"}), 413


@app.get("/health")
def health():
    loaded = load_model() is not None
    payload = {"status": "ok" if loaded else "degraded", "model_loaded": loaded}
    if not loaded:
        payload["error"] = model_error
    return jsonify(payload), 200 if loaded else 503


@app.post("/predict")
def predict():
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "No file uploaded. Send an image using the 'file' field."}), 400
    if load_model() is None:
        return jsonify({"error": model_error}), 503

    try:
        processed = preprocess_image(request.files["file"])
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    probabilities = model.predict(processed, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    return jsonify({
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 4),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)