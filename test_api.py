import io

import requests
from PIL import Image


API_URL = "http://127.0.0.1:5000"


def main():
    health = requests.get(f"{API_URL}/health", timeout=5)
    print("GET /health", health.status_code, health.json())

    missing_file = requests.post(f"{API_URL}/predict", timeout=5)
    print("POST /predict without file", missing_file.status_code, missing_file.json())

    image = Image.new("L", (28, 28), color=0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    prediction = requests.post(
        f"{API_URL}/predict",
        files={"file": ("sample_digit.png", buffer.getvalue(), "image/png")},
        timeout=30,
    )
    print("POST /predict", prediction.status_code, prediction.json())


if __name__ == "__main__":
    main()