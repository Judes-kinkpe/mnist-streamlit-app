import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ================================
# CONFIG PAGE
# ================================
st.set_page_config(
    page_title="Projet de reconnaissance de chiffre manuscrit",
    page_icon="✏️",
    layout="centered"
)

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.keras")

model = load_model()

# ================================
# PREPROCESSING
# ================================
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    _, img = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, 0.0

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    quality = min(1.0, cv2.contourArea(cnt) / 500)

    digit = img[y:y+h, x:x+w]
    digit = cv2.resize(digit, (20, 20))

    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[4:24, 4:24] = digit

    padded = padded / 255.0
    padded = padded.reshape(1, 28, 28, 1)

    return padded, quality

# ================================
# TTA PREDICTION
# ================================
def predict_with_tta(img, n=5):
    preds = []

    for _ in range(n):
        shift_x = np.random.randint(-2, 2)
        shift_y = np.random.randint(-2, 2)

        shifted = np.roll(img, shift_x, axis=1)
        shifted = np.roll(shifted, shift_y, axis=2)

        preds.append(model.predict(shifted, verbose=0))

    return np.mean(preds, axis=0)

# ================================
# UI
# ================================
st.title(" Reconnaissance de chiffres manuscrits")
st.markdown(
    "Téléverse une **image manuscrite** d’un chiffre entre **0 et 9**."
)

# ================================
# UPLOAD IMAGE
# ================================
uploaded = st.file_uploader(
    "📷 Choisir une image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Image téléchargée", width=200)

    img = np.array(image)
    processed, quality = preprocess_image(img)

    if processed is not None:
        st.subheader("Résultat")

        prediction = predict_with_tta(processed)
        digit = np.argmax(prediction)

        st.success(f" Le chiffre reconnu est : **{digit}**")

        st.metric(
            "Score qualité du prétraitement",
            f"{quality*100:.1f} %"
        )

        st.subheader("📊 Probabilités par chiffre")
        probs = prediction[0]

        for i, p in enumerate(probs):
            st.progress(float(p), text=f"{i} : {p*100:.1f}%")

    else:
        st.error(" Aucun chiffre valide détecté.")

else:
    st.info("📤 Téléverse une image pour lancer la reconnaissance.")
