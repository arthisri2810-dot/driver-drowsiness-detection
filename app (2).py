import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗")

st.title("🧠 Driver Drowsiness Detection System")

@st.cache_resource
def load_my_model():
    return load_model("driver_drowsiness_mobilenet_cpu.keras")

model = load_my_model()

CLASS_NAMES = ['Closed', 'Open', 'no_yawn', 'yawn']

def fatigue_level(pred_class):
    if pred_class in ['Open','no_yawn']:
        return "🟢 Alert"
    elif pred_class == 'yawn':
        return "🟡 Mild Fatigue"
    else:
        return "🔴 Severe Fatigue"

uploaded_file = st.file_uploader("Upload Driver Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img)

    img = img.resize((224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_class = CLASS_NAMES[pred_index]
    confidence = np.max(prediction)*100

    st.subheader("Prediction")
    st.write("Class:", pred_class)
    st.write("Confidence: {:.2f}%".format(confidence))
    st.write("Fatigue Level:", fatigue_level(pred_class))

    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, prediction[0])
    ax.set_ylim(0,1)
    st.pyplot(fig)
