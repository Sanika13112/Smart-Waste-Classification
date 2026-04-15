import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from streamlit_lottie import st_lottie
import requests

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Smart Waste AI", page_icon="♻️", layout="wide")

# ---------------- LOAD LOTTIE ---------------- #
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottie("https://assets2.lottiefiles.com/packages/lf20_kyu7xb1v.json")

# ---------------- CSS ---------------- #
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cfd8dc;
}

.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px;
    margin-top: 20px;
}

.result {
    font-size: 28px;
    font-weight: bold;
    text-align: center;
}

.info {
    text-align: center;
    color: #ccc;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ---------------- #
model_binary = load_model(r"C:\Users\Sanika\Downloads\EVS_Project\model1.keras")
model_multi = load_model(r"C:\Users\Sanika\Downloads\Garbage _Classification\model.keras")

binary_classes = ['Organic 🌿', 'Recyclable ♻️']

multi_classes = [
    'Plastic 🧴',
    'Cardboard 📦',
    'Metal 🥫',
    'Trash 🚮',
    'Paper 📄',
    'Glass 🍾'
]

# ---------------- DESCRIPTIONS ---------------- #
descriptions = {
    'Organic 🌿': "Biodegradable waste like food and plants.",
    'Recyclable ♻️': "Waste that can be reused or processed.",
    'Plastic 🧴': "Used in bottles and packaging.",
    'Cardboard 📦': "Paper-based packaging material.",
    'Metal 🥫': "Aluminum or steel waste.",
    'Trash 🚮': "Non-recyclable general waste.",
    'Paper 📄': "Reusable paper materials.",
    'Glass 🍾': "Recyclable glass containers."
}

# ---------------- HEADER ---------------- #
col1, col2 = st.columns([3,1])

with col1:
    st.markdown('<div class="title">♻️ Smart Waste Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered intelligent waste detection system</div>', unsafe_allow_html=True)

with col2:
    st_lottie(lottie_ai, height=150)

st.markdown("---")

# ---------------- UPLOAD ---------------- #
file = st.file_uploader("📤 Upload Waste Image", type=["jpg","png","jpeg"])

if file:

    image = Image.open(file).convert("RGB")

    col1, col2 = st.columns(2)

    # IMAGE
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # RESULT
    with col2:

        img = image.resize((224,224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        # -------- STAGE 1 -------- #
        pred1 = model_binary.predict(img_array)
        class1 = binary_classes[np.argmax(pred1)]
        conf1 = np.max(pred1)*100

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Stage 1")
        st.markdown(f'<div class="result">{class1}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info">{descriptions[class1]}</div>', unsafe_allow_html=True)
        st.progress(int(conf1))
        st.write(f"Confidence: {conf1:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        # -------- STAGE 2 -------- #
        if "Recyclable" in class1:

            pred2 = model_multi.predict(img_array)
            class2 = multi_classes[np.argmax(pred2)]
            conf2 = np.max(pred2)*100

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🧠 Detailed Classification")
            st.markdown(f'<div class="result">{class2}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info">{descriptions[class2]}</div>', unsafe_allow_html=True)
            st.progress(int(conf2))
            st.write(f"Confidence: {conf2:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#aaa;'>🌍 AI for Smart & Sustainable Cities</p>", unsafe_allow_html=True)