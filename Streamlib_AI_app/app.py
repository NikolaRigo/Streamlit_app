import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd

# --- 1. CONFIGURATION ---
CLASS_NAMES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Pothole"]

SEVERITY_ORDER = ["Pothole", "Alligator Crack", "Transverse Crack", "Longitudinal Crack"]

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

def predict_single_image(model, img):
    input_tensor = process_image(img)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    all_probs = probabilities.tolist()
    
    # 0: Longitudinal, 1: Transverse, 2: Alligator, 3: Pothole
    p_pothole = probabilities[3].item()
    p_alligator = probabilities[2].item()
    
    # Standard Prediction
    max_conf, max_pred_idx = torch.max(probabilities, 0)
    final_pred_idx = max_pred_idx.item()
    severity_override = False

    # --- SAFETY LOGIC: POTHOLE OVERRIDE ---
    if p_pothole >= 0.20:
        final_pred_idx = 3  # Force Pothole
        severity_override = True
        final_conf = p_pothole # Use the pothole probability as the confidence displayed
    elif p_alligator >= 0.30:
        final_pred_idx = 2  # Force Alligator Crack
        severity_override = True
        final_conf = p_alligator
    else:
        final_conf = max_conf.item()

    return CLASS_NAMES[final_pred_idx], final_conf, severity_override, all_probs


# --- 4. THE USER INTERFACE ---
st.title("🛣️ RoadGuard Damage Detector")
st.write(f"Running on: **{DEVICE.upper()}** (RTX 3050 detected)" if DEVICE == "cuda" else "Running on: **CPU**")

model = load_model()

if model is None:
    st.error(f"⚠️ Could not find '{MODEL_PATH}'. Make sure your training finished and the file is in this folder!")
else:
    uploaded_files = st.file_uploader("Upload a photo of a road surface...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.info(f"Uploaded {len(uploaded_files)} image(s). Ready to initiate classification.")
        
        if st.button("🔍 Analyze Road Surface"):
            grouped_results = {k: [] for k in CLASS_NAMES}

            progress_bar = st.progress(0)

            for i, file in enumerate(uploaded_files):
                img = Image.open(file).convert("RGB")
                
                label, conf, override, all_probs = predict_single_image(model, img)

                grouped_results[label].append({
                    "image": img,
                    "filename": file.name,
                    "confidence": conf,
                    "override": override,
                    "all_probs": all_probs
                })

                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success("Analysis complete! See results below.")
            st.divider()

            for category in SEVERITY_ORDER:
                items = grouped_results[category]

                if items:
                    if category == "Pothole":
                        st.markdown("## 🚨 Potholes Detected")
                        st.markdown("These require immediate attention due to safety risks.")
                    elif category == "Alligator Crack":
                        st.markdown("## ⚠️ Alligator Cracks Detected")
                        st.markdown("These indicate severe road damage needing prompt repair.")
                    else:
                        st.markdown(f"## ℹ️ {category}s Detected")
                        st.markdown("These are less severe but should be monitored.")

                    
                    cols = st.columns(3)
                    for idx, item in enumerate(items):
                        with cols[idx % 3]:
                            st.image(item["image"], use_container_width=True)
                            
                            if item["override"]:
                                st.caption(f"**⚠️{item['filename']}**")
                                st.error(f"⚠️ Risk: {item['confidence']*100:.1f}% (Override Active)")
                            else:
                                st.caption(f"{item['filename']}")
                                st.write(f"Confidence: {item['confidence']*100:.1f}%")

                            table_data = {
                                "Defect Type": CLASS_NAMES,
                                "Confidence (%)": [f"{p*100:.1f}%" for p in item["all_probs"]]
                            }
                            df = pd.DataFrame(table_data)

                            st.dataframe(
                                df,
                                hide_index=True,
                                use_container_width=True,
                                height=150
                            )

                    st.divider()

# --- SIDEBAR INFO ---
st.sidebar.info(
    "### About This Model\nThis AI was trained to 74% accuracy using MobileNetV2. It specializes in identifying Cracks and Potholes.")