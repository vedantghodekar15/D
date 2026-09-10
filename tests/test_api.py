import io

from PIL import Image

from app import app, preprocess_image


def test_health_reports_api_state():
    response = app.test_client().get("/health")

    assert response.status_code in (200, 503)
    assert response.json["status"] in ("ok", "degraded")
    assert isinstance(response.json["model_loaded"], bool)


def test_predict_requires_a_file():
    response = app.test_client().post("/predict")

    assert response.status_code == 400
    assert "No file uploaded" in response.json["error"]


def test_preprocess_image_returns_mnist_shape():
    image = Image.new("L", (100, 100), color=255)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    processed = preprocess_image(buffer)

    assert processed.shape == (1, 784)
    assert processed.dtype == "float32"
    assert float(processed.min()) >= 0.0
    assert float(processed.max()) <= 1.0