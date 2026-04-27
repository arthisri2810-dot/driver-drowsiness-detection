import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import mediapipe as mp

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗")
st.title("🚗 Driver Drowsiness Detection")

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    return load_model("eye_mobilenet.keras"), load_model("mouth_mobilenet.keras")

eye_model, mouth_model = load_models()

eye_classes = ['closed', 'open']
mouth_classes = ['no_yawn', 'yawn']

# ==============================
# MEDIAPIPE
# ==============================
@st.cache_resource
def load_mp():
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

face_mesh = load_mp()

# ==============================
# PREPROCESS
# ==============================
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ==============================
# EYE VISIBILITY
# ==============================
def is_eye_visible(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.var(gray) > 50

# ==============================
# NORMALIZED MOUTH RATIO
# ==============================
def mouth_open_ratio(lm):
    top = lm[13]
    bottom = lm[14]
    left = lm[78]
    right = lm[308]

    vertical = abs(bottom.y - top.y)
    horizontal = abs(right.x - left.x)

    if horizontal == 0:
        return 0

    return vertical / horizontal

# ==============================
# GET REGIONS
# ==============================
def get_regions(img):
    h, w, _ = img.shape
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if not results.multi_face_landmarks:
        return None, None, None

    lm = results.multi_face_landmarks[0].landmark

    def crop(indices, pad=25):
        xs = [int(lm[i].x * w) for i in indices]
        ys = [int(lm[i].y * h) for i in indices]

        x1, x2 = max(0, min(xs)-pad), min(w, max(xs)+pad)
        y1, y2 = max(0, min(ys)-pad), min(h, max(ys)+pad)

        crop = img[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    eye_idx = [33,133,160,159,158,157,173]
    mouth_idx = [13,14,78,308]

    return crop(eye_idx), crop(mouth_idx), lm

# ==============================
# FINAL FATIGUE LOGIC
# ==============================
def fatigue_logic(eye_label, mouth_label, eye_conf):

    if eye_label == "Unknown" or eye_conf < 0.80:
        return "Mild Fatigue" if mouth_label == "Yawn" else "Alert"

    if eye_label == "Open" and mouth_label == "No Yawn":
        return "Alert"

    if eye_label == "Open" and mouth_label == "Yawn":
        return "Mild Fatigue"

    if eye_label == "Closed" and mouth_label == "Yawn":
        return "Severe Fatigue"

    if eye_label == "Closed":
        return "Moderate Fatigue"

    return "Moderate Fatigue"

# ==============================
# UPLOAD
# ==============================
file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:

    img = np.array(Image.open(file).convert("RGB"))
    st.image(img, use_container_width=True)

    eye_img, mouth_img, lm = get_regions(img)

    if eye_img is None or mouth_img is None:
        st.warning("⚠ Face not detected. Using full image.")
        eye_img, mouth_img = img, img
        use_geo = False
    else:
        use_geo = True

    # ==============================
    # EYE
    # ==============================
    if not is_eye_visible(eye_img):
        eye_label = "Unknown"
        eye_conf = 0.0
    else:
        ep = eye_model.predict(preprocess(eye_img), verbose=0)[0]
        ei = np.argmax(ep)
        eye_conf = float(np.max(ep))
        eye_label = eye_classes[ei].title()

    # ==============================
    # MOUTH (🔥 FINAL FIX)
    # ==============================
    mpred = mouth_model.predict(preprocess(mouth_img), verbose=0)[0]
    mi = np.argmax(mpred)
    mouth_conf = float(np.max(mpred))

    if use_geo:
        ratio = mouth_open_ratio(lm)

        if ratio > 0.32 and mouth_conf > 0.80:
            mouth_label = "Yawn"
        elif ratio > 0.40:
            mouth_label = "Yawn"
        else:
            mouth_label = "No Yawn"
    else:
        mouth_label = mouth_classes[mi].replace("_"," ").title()

    # ==============================
    # FATIGUE
    # ==============================
    fatigue = fatigue_logic(eye_label, mouth_label, eye_conf)

    st.write("---")

    c1, c2 = st.columns(2)
    c1.metric("👁 Eye", eye_label, f"{eye_conf:.2f}")
    c2.metric("👄 Mouth", mouth_label, f"{mouth_conf:.2f}")

    st.write("---")

    # ==============================
    # ALERT
    # ==============================
    if fatigue == "Severe Fatigue":
        st.error("🚨 Severe Fatigue Detected!")
    elif fatigue == "Moderate Fatigue":
        st.warning("⚠ Moderate Fatigue")
    elif fatigue == "Mild Fatigue":
        st.info("😴 Mild Fatigue")
    else:
        st.success("✅ Alert")

    st.write("### Confidence")
    st.progress(int(eye_conf*100))
    st.progress(int(mouth_conf*100))
