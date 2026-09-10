# Experiment 8: Model Deployment (API + UI)

This project deploys an MNIST feed-forward neural network through a Flask REST API and a Streamlit UI.

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the model

The training script downloads MNIST, trains the baseline ANN, evaluates it, and saves `models/baseline_ann.keras`.

```bash
python train_model.py
```

## Run the application

Open two terminals in the project directory:

```bash
python app.py
streamlit run ui.py
```

Open the Streamlit URL printed in the terminal, upload a digit image, and select **Predict digit**.

## Test the API

With the Flask server running:

```bash
python test_api.py
curl -X POST http://127.0.0.1:5000/predict -F "file=@sample_digit.png"
```

Endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Reports API and model readiness |
| `/predict` | POST | Accepts an image in the `file` form field and returns digit plus confidence |

The API converts uploaded images to grayscale, automatically handles dark ink on a light background, crops and centers the digit on a 28x28 canvas, and scales values to `[0, 1]` to match MNIST training preprocessing.

## Experiment 9: MLOps basics

This project includes Git-ready version control, a Docker image for the Flask API, automated API tests, and a GitHub Actions workflow. TensorFlow is pinned to 2.21.0 so the training and serving runtimes use the same Keras model format. The Docker image installs only API dependencies; Streamlit remains available through `requirements.txt` for the UI.

### Git version control

Run these commands once from the project directory:

```bash
git init
git add .
git commit -m "Initial commit: MLOps deployment pipeline"
git branch -M main
```

Add your GitHub remote and push when the repository has been created:

```bash
git remote add origin https://github.com/<username>/mlops-basics.git
git push -u origin main
```

The model file is ignored by Git because generated model binaries should normally be stored in a model registry or tracked with DVC rather than committed directly.

### Run tests

```bash
python -m pytest tests/ -q
```

### Build and run Docker

Make sure Docker Desktop is running, then execute:

```bash
docker build -t dl-lab-app:latest .
docker run --rm -p 5000:5000 dl-lab-app:latest
```

Verify the container from another terminal:

```bash
curl http://localhost:5000/health
```

The Docker image includes the local model when `models/baseline_ann.keras` is present. A fresh clone must run `python train_model.py` before starting the API, or retrieve the model from the configured artifact store.

### CI/CD

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`. It installs the requirements, runs the API tests, and builds the Docker image. GitHub Actions requires the model artifact to be supplied separately for a fully healthy inference container; the build and contract tests do not depend on that binary.