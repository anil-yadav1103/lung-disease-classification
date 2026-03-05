import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Model file name
MODEL_FILE = "efficientnetv2m_final_model.keras"

# Google Drive file ID
FILE_ID = "1y2ityFxupLlsaU4CRTRfyiGE_SwITntO"

# Download model if not present
if not os.path.exists(MODEL_FILE):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(MODEL_FILE, compile=False)

# Streamlit page title
st.title("Lung Disease Detection using Chest X-ray")

st.write("Upload a chest X-ray image to detect Pneumonia.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (same as training)
    image = image.resize((256, 256))
    img = np.array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    # Binary classification
    if prediction[0][0] > 0.5:
        diagnosis = "PNEUMONIA"
        confidence = float(prediction[0][0])
    else:
        diagnosis = "NORMAL"
        confidence = float(1 - prediction[0][0])

    # Display result
    st.subheader("Prediction Result")
    st.write(f"Diagnosis: **{diagnosis}**")
    st.write(f"Confidence: **{confidence:.2f}**")
