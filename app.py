import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from huggingface_hub import hf_hub_download

# ── Download model from Hugging Face Hub (reliable, no quota limits) ──────────
MODEL_FILE = "efficientnetv2m_final_model.keras"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Anilbommanoni/chest-xray-model",   # ← your HF model repo name
        filename=MODEL_FILE,
        repo_type="model"
    )
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("Lung Disease Detection using Chest X-ray")
st.write("Upload a chest X-ray image to detect Pneumonia.")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess (same as training)
    image = image.resize((256, 256))
    img = np.array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        diagnosis  = "PNEUMONIA"
        confidence = float(prediction[0][0])
    else:
        diagnosis  = "NORMAL"
        confidence = float(1 - prediction[0][0])

    st.subheader("Prediction Result")
    st.write(f"Diagnosis: **{diagnosis}**")
    st.write(f"Confidence: **{confidence:.2f}**")
