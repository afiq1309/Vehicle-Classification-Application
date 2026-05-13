import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
import tensorflow as tf

# =========================
# CONFIG (EDIT THESE 3)
# =========================
MODEL_PATH = Path("runs/EfficientNetB0_frozen/best.keras")
DATASET_DIR = Path("C:/Users/ASUS/Desktop/Afiq/Study/VIP/Project/Classification_Dataset")
IMG_SIZE = (128, 128)  # MUST match training image_size

st.set_page_config(page_title="Vehicle Classifier", layout="centered")
st.title("Vehicle Classification (EfficientNet)")

# =========================
# HELPERS
# =========================
def get_class_names_from_dataset(dataset_dir: Path) -> list[str]:
    """Read class names from dataset/train/<class_name>/"""
    train_dir = dataset_dir / "train"
    if not train_dir.exists():
        return []
    return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

def preprocess_for_model(img: Image.Image) -> np.ndarray:
    """Resize + convert to model input array."""
    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img).astype(np.float32)  # (H, W, 3), values 0..255
    x = np.expand_dims(x, axis=0)         # (1, H, W, 3)
    return x

@st.cache_resource
def load_model():
    """Load trained Keras model once and cache it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")
    return tf.keras.models.load_model(MODEL_PATH)

# =========================
# CLASS NAMES CHECK
# =========================
class_names = get_class_names_from_dataset(DATASET_DIR)

if not class_names:
    st.error(
        "Could not find class folders in dataset.\n\n"
        f"Expected: {DATASET_DIR}/train/<class_name>/\n"
        "Fix DATASET_DIR in app.py."
    )
    st.stop()

st.caption(f"Classes: {class_names}")

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

classify_clicked = st.button("Classify vehicle", use_container_width=True)

# =========================
# DISPLAY + PREDICT
# =========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if classify_clicked:
        try:
            model = load_model()
            x = preprocess_for_model(img)
            prob = float(model.predict(x, verbose=0).reshape(-1)[0])

            # Binary sigmoid output: class 1 if prob >= 0.5 else class 0
            pred_idx = 1 if prob >= 0.5 else 0
            pred_name = class_names[pred_idx]

            # Confidence for predicted class
            confidence = prob if pred_idx == 1 else (1.0 - prob)

            st.subheader("Prediction")
            st.write(f"**Class:** {pred_name}")
            st.write(f"**Confidence:** {confidence:.4f}")

        except Exception as e:
            st.error(f"Classification failed: {e}")
else:
    st.info("Please upload an image to classify.")
