import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

MODEL_FILE = "efficientnetv2m_final_model.keras"

# Download model from Google Drive if not present
if not os.path.exists(MODEL_FILE):
    file_id = "1y2ityFxupLlsaU4CRTRfyiGE_SwITntO"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_FILE, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_FILE, compile=False)

# Class labels
class_names = ["Normal", "Pneumonia"]

st.title("Lung Disease Detection using Chest X-ray")

st.write("Upload a chest X-ray image to detect lung disease.")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = img.reshape(1,224,224,3)

    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader("Prediction Result")
    st.write(f"Diagnosis: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}**")
