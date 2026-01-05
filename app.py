import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ======================
# Chargement du modèle
# ======================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.keras")

model = load_model()

# ======================
# Prétraitement MNIST
# ======================
def preprocess_image_for_mnist(pil_image):
    """
    Prétraitement d'une image réelle pour la rendre compatible MNIST
    Sortie : image (1, 28, 28, 1)
    """

    img = np.array(pil_image)

    # Sécurité : conversion en niveaux de gris
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Réduction du bruit
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Binarisation adaptative (fond clair/sombre)
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Nettoyage morphologique
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Détection du chiffre (plus grand contour)
    contours, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    if w < 10 or h < 10:
        return None

    digit = img[y:y+h, x:x+w]

    # Redimensionnement avec padding
    digit = cv2.resize(digit, (20, 20))
    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[4:24, 4:24] = digit

    # Normalisation
    padded = padded / 255.0
    padded = padded.reshape(1, 28, 28, 1)

    return padded

# ======================
# Interface Streamlit
# ======================
st.title("Projet de reconnaissance des chiffres manuscrits")
st.write("Téléverse une image manuscrite d’un chiffre (0 à 9).")

uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Image téléchargée", width=150)

    processed_img = preprocess_image_for_mnist(image)

    if processed_img is None:
        st.error("Aucun chiffre valide détecté.")
    else:
        prediction = model.predict(processed_img)
        digit = np.argmax(prediction)
        st.success(f"Le chiffre reconnu est : {digit}")