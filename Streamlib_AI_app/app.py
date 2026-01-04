import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os

# --- 1. CONFIGURATION ---
CLASS_NAMES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Pothole"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_road_model.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="RoadGuard AI", page_icon="🛣️")


# --- 2. MODEL LOADING ---
@st.cache_resource  # Keep the model in memory so it doesn't reload every click
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    else:
        return None


# --- 3. IMAGE PREPROCESSING ---
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


# --- 4. THE USER INTERFACE ---
st.title("🛣️ RoadGuard Damage Detector")
st.write(f"Running on: **{DEVICE.upper()}** (RTX 3050 detected)" if DEVICE == "cuda" else "Running on: **CPU**")

model = load_model()

if model is None:
    st.error(f"⚠️ Could not find '{MODEL_PATH}'. Make sure your training finished and the file is in this folder!")
else:
    uploaded_file = st.file_uploader("Upload a photo of a road surface...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display the image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyze Road Surface"):
            with st.spinner("Analyzing pixels..."):
                # Run prediction
                input_tensor = process_image(img)
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    conf, pred = torch.max(probabilities, 0)

                # Show results
                label = CLASS_NAMES[pred.item()]
                confidence_score = conf.item() * 100

                st.divider()
                st.subheader(f"Result: {label}")

                # Visual feedback based on damage type
                if label == "Pothole":
                    st.error(f"Critical: {label} detected. Repair recommended.")
                else:
                    st.warning(f"Maintenance: {label} detected.")

                st.write(f"**Confidence:** {confidence_score:.2f}%")
                st.progress(conf.item())

                # Show all class probabilities for transparency
                with st.expander("See Detailed Probability Breakdown"):
                    for i, prob in enumerate(probabilities):
                        st.write(f"{CLASS_NAMES[i]}: {prob.item() * 100:.1f}%")

# --- SIDEBAR INFO ---
st.sidebar.info(
    "### About This Model\nThis AI was trained to 74% accuracy using MobileNetV2. It specializes in identifying Cracks and Potholes.")