import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:5000")

st.set_page_config(page_title="Digit Recognition")
st.title("Digit Recognition")
st.caption("Upload a handwritten digit and ask the deployed ANN model to classify it.")

try:
    health_response = requests.get(f"{API_URL}/health", timeout=3)
    if health_response.ok:
        st.success("API is online")
    else:
        st.warning("API is online, but the model is not ready. Run train_model.py first.")
except requests.RequestException:
    st.error(f"Cannot reach the API at {API_URL}. Start it with: python app.py")

uploaded_file = st.file_uploader("Choose a PNG or JPG image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", width=180)
    if st.button("Predict digit", type="primary"):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                timeout=30,
            )
            result = response.json()
            if response.ok:
                st.metric("Predicted digit", result["predicted_digit"])
                st.write(f"Confidence: {result['confidence'] * 100:.2f}%")
            else:
                st.error(result.get("error", "Prediction failed."))
        except (requests.RequestException, ValueError):
            st.error("Prediction failed. Check that the Flask API is running.")