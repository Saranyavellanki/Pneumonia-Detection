import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title=" Pneumonia Detection",
    layout="centered"
)

# ------------------------------
# TITLE
# ------------------------------
st.title("🫁 Pneumonia Detection System")
st.write("Upload a chest X-ray image and the model will predict if pneumonia is present.")

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource  # caches model so it loads only once
def load_model():
    model = tf.keras.models.load_model("model/pneumonia_model.h5")
    return model

model = load_model()

# ------------------------------
# UPLOAD IMAGE
# ------------------------------
uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess for model
    IMG_SIZE = 224
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    # Predict
    pred = model.predict(img_array)
    score = float(pred[0][0])

    # Display result
    if score > 0.5:
        st.error(f"🚨 Pneumonia Detected! (Confidence: {score:.2f})")
    else:
        st.success(f"✅ Normal Chest X-ray (Confidence: {1-score:.2f})")

    st.info("⚠️ This tool is for educational purposes only and is not a medical diagnosis.")
