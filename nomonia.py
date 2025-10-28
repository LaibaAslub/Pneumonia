import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# 🧠 Load trained model
model = tf.keras.models.load_model("pneumonia_model.keras")

# 🔹 Get input shape dynamically
input_shape = model.input_shape[1:4]

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="centered")

st.title("🫁 Pneumonia Detection App")
st.write("Upload a **chest X-ray image** to predict whether it shows **Pneumonia (Infected)** or **Normal (Uninfected)**.")

# 📤 Image uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # ✅ Open and fix orientation
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.exif_transpose(image)
    st.image(image, caption="🩺 Uploaded X-ray", use_container_width=True)

    # ✅ Preprocess image
    img = image.resize((input_shape[1], input_shape[0]))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, h, w, 1)

    # ✅ Predict
    prediction = model.predict(img_array)
    prob = prediction[0][0]

    # ✅ Interpret output
    if prob > 0.5:
        result = "🦠 Pneumonia Detected (Infected)"
        confidence = prob * 100
        st.error(f"**Prediction:** {result}")
    else:
        result = "✅ Normal (Uninfected)"
        confidence = (1 - prob) * 100
        st.success(f"**Prediction:** {result}")

    # ✅ Show confidence
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.caption(f"Raw model output: {prob:.4f}")
