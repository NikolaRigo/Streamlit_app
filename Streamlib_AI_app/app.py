import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# --- 1. CONFIGURATION ---
CLASS_NAMES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Pothole"]

SEVERITY_ORDER = ["Pothole", "Alligator Crack", "Transverse Crack", "Longitudinal Crack"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_Stage2.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="RoadGuard AI", page_icon="🛣️", layout="wide")

# Custom CSS for better image display
st.markdown("""
<style>
    /* Force images to display at full width */
    .stImage img {
        width: 100% !important;
        height: auto !important;
        object-fit: contain;
    }
    
    /* Improve card spacing */
    [data-testid="stContainer"] {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. MODEL LOADING ---
@st.cache_resource  # Keep the model in memory so it doesn't reload every click
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 4)  # NUM_CLASSES
    )

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
        transforms.ToTensor()
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


# --- 4. EXPORT FUNCTIONS ---

def generate_pdf_report(grouped_results, analysis_time):
    """Generate a comprehensive PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    elements.append(Paragraph("🛣️ RoadGuard AI Analysis Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Metadata
    meta_data = [
        ['Analysis Date:', analysis_time],
        ['Total Images Analyzed:', str(sum(len(v) for v in grouped_results.values()))],
        ['Device Used:', DEVICE.upper()],
        ['Model:', 'MobileNetV2 (74% accuracy)']
    ]
    
    meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 20))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", heading_style))
    
    summary_data = [['Defect Type', 'Count', 'Severity Level']]
    severity_map = {
        'Pothole': 'CRITICAL',
        'Alligator Crack': 'HIGH',
        'Transverse Crack': 'MEDIUM',
        'Longitudinal Crack': 'LOW'
    }
    
    for category in SEVERITY_ORDER:
        count = len(grouped_results[category])
        if count > 0:
            summary_data.append([category, str(count), severity_map[category]])
    
    summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    
    # Detailed Results
    elements.append(Paragraph("Detailed Analysis Results", heading_style))
    elements.append(Spacer(1, 12))
    
    for category in SEVERITY_ORDER:
        items = grouped_results[category]
        if items:
            # Category header
            category_style = ParagraphStyle(
                'Category',
                parent=styles['Heading3'],
                fontSize=14,
                textColor=colors.HexColor('#e74c3c') if category == 'Pothole' else colors.HexColor('#34495e'),
                spaceAfter=8
            )
            elements.append(Paragraph(f"{category} ({len(items)} detected)", category_style))
            
            # Create table for this category
            table_data = [['Filename', 'Confidence', 'Override', 'Long.', 'Trans.', 'Allig.', 'Pothole']]
            
            for item in items:
                table_data.append([
                    item['filename'][:25] + '...' if len(item['filename']) > 25 else item['filename'],
                    f"{item['confidence']*100:.1f}%",
                    '⚠️' if item['override'] else '✓',
                    f"{item['all_probs'][0]*100:.0f}%",
                    f"{item['all_probs'][1]*100:.0f}%",
                    f"{item['all_probs'][2]*100:.0f}%",
                    f"{item['all_probs'][3]*100:.0f}%"
                ])
            
            detail_table = Table(table_data, colWidths=[2*inch, 0.8*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.7*inch])
            detail_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            elements.append(detail_table)
            elements.append(Spacer(1, 15))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


# --- 5. THE USER INTERFACE ---
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
            analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
            
            # Store results in session state for export
            st.session_state['grouped_results'] = grouped_results
            st.session_state['analysis_time'] = analysis_time
            
            st.success("Analysis complete! See results below.")
            
            # --- EXPORT BUTTONS ---
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # PDF Export
                pdf_buffer = generate_pdf_report(grouped_results, analysis_time)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"roadguard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with col2:
                total_images = sum(len(v) for v in grouped_results.values())
                critical_count = len(grouped_results["Pothole"]) + len(grouped_results["Alligator Crack"])
                st.metric("Total Analyzed", total_images)
                st.metric("Critical Issues", critical_count, delta="High Priority" if critical_count > 0 else None)
            
            st.divider()

            # --- DISPLAY RESULTS WITH CARD-BASED LAYOUT ---
            for category in SEVERITY_ORDER:
                items = grouped_results[category]

                if items:
                    # Category header with emoji and description
                    if category == "Pothole":
                        st.markdown("## 🚨 Potholes Detected")
                        st.markdown("These require immediate attention due to safety risks.")
                    elif category == "Alligator Crack":
                        st.markdown("## ⚠️ Alligator Cracks Detected")
                        st.markdown("These indicate severe road damage needing prompt repair.")
                    else:
                        st.markdown(f"## ℹ️ {category}s Detected")
                        st.markdown("These are less severe but should be monitored.")

                    # Create cards in 3-column layout
                    cols = st.columns(3)
                    for idx, item in enumerate(items):
                        with cols[idx % 3]:
                            # Card container
                            with st.container(border=True):
                                # Image with fixed height
                                st.image(item["image"], use_container_width=True, output_format="JPEG")
                                
                                # Add some spacing
                                st.markdown("")
                                
                                # Title (filename)
                                st.markdown(f"**{item['filename']}**")
                                
                                # Severity badge with emoji
                                severity_badges = {
                                    "Pothole": "🔴 POTHOLE",
                                    "Alligator Crack": "🟠 ALLIGATOR CRACK",
                                    "Transverse Crack": "🟡 TRANSVERSE CRACK",
                                    "Longitudinal Crack": "🔵 LONGITUDINAL CRACK"
                                }
                                st.markdown(f"### {severity_badges[category]}")
                                
                                # Confidence metric
                                st.metric(
                                    label="Confidence",
                                    value=f"{item['confidence']*100:.1f}%",
                                    delta="High" if item['confidence'] > 0.7 else ("Medium" if item['confidence'] > 0.4 else "Low")
                                )
                                
                                # Override warning
                                if item["override"]:
                                    st.warning("⚠️ Safety Override Applied", icon="⚠️")
                                
                                # Expandable detailed probabilities
                                with st.expander("📊 Show Detailed Probabilities"):
                                    st.markdown("**All Class Probabilities:**")
                                    
                                    # Create horizontal bars for each class
                                    for class_idx, class_name in enumerate(CLASS_NAMES):
                                        prob_value = item['all_probs'][class_idx]
                                        st.progress(prob_value, text=f"{class_name}: {prob_value*100:.1f}%")
                                    
                                    # Additional metadata
                                    st.divider()
                                    st.caption(f"Analyzed: {analysis_time}")

                    st.divider()

# --- SIDEBAR INFO ---
st.sidebar.info(
    "### About This Model\n"
    "This AI was trained to 74% accuracy using MobileNetV2. "
    "It specializes in identifying Cracks and Potholes.\n\n"
    "### Features\n"
    "✅ Card-based layout\n"
    "✅ Severity badges with color coding\n"
    "✅ Confidence metrics\n"
    "✅ Expandable probability details\n\n"
    "### Export Options\n"
    "After analysis, you can download:\n"
    "- **PDF Report**: Professional summary with statistics"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Version:** 2.1 (Card Layout)")