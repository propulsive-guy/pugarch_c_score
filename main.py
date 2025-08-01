import streamlit as st
from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
from PIL import Image
import tempfile
import os

# Set up the Streamlit page
st.set_page_config(page_title="Cleanliness Score Dashboard", page_icon="🧹", layout="centered")

# Header and instructions
st.markdown("""
    <h1 style="text-align:center; color:#88C0D0;">🧼 Cleanliness Score Dashboard</h1>
    <p style="text-align:center; color:#D8DEE9;">Upload an image and get cleanliness level prediction using YOLOv8</p>
    <hr>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name

    st.image(image_path, caption="📷 Uploaded Image", use_column_width=True)

    # Load YOLO model (ensure 'best.pt' is in same directory or update path accordingly)
    model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    model = YOLO(model_path)

    # Perform inference
    results = model(image_path)[0]

    # Extract class IDs and confidence scores
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()

    # Mapping class IDs to readable names
    class_mapping = {
        0: 'fine dust',
        1: 'garbagebag',
        2: 'liquid',
        3: 'paper_waste',
        4: 'plastic_bottles',
        5: 'plasticbags',
        6: 'stains'
    }

    # Define raw weights for each class (higher = dirtier)
    original_weights = {
        0: 1,
        1: 5,
        2: 4,
        3: 2,
        4: 3,
        5: 4,
        6: 3
    }

    # Normalize weights to sum to 10
    total_weight = sum(original_weights.values())
    normalized_weights = {cls: (wt / total_weight * 10) for cls, wt in original_weights.items()}

    # Calculate score
    raw_score = 0
    class_confidence_dict = defaultdict(list)

    for cls_id, conf in zip(class_ids, confidences):
        weight = normalized_weights.get(cls_id, 1)
        raw_score += weight * conf
        class_confidence_dict[cls_id].append(conf)

    cleanliness_score = max(0, round(10 - raw_score, 2))

    # Display breakdown
    st.markdown("### 📊 Class-wise Detection Summary")
    for cls_id, conf_list in class_confidence_dict.items():
        count = len(conf_list)
        avg_conf = np.mean(conf_list)
        st.write(f"- **{class_mapping[cls_id]}**: {count} × avg conf {round(avg_conf, 2)} × weight {round(normalized_weights[cls_id], 2)}")

    # Display scores
    st.markdown(f"### 💥 Raw Dirtiness Score: **{round(raw_score, 2)}**")
    st.markdown(f"### 🧼 Final Cleanliness Score (0 = dirtiest, 10 = cleanest): **{cleanliness_score} / 10**")

    # Generate prediction image
    with tempfile.TemporaryDirectory() as pred_dir:
        model.predict(
            image_path,
            save=True,
            save_txt=False,
            save_conf=True,
            project=pred_dir,
            name="result",
            exist_ok=True
        )
        result_dir = os.path.join(pred_dir, "result")
        output_image_path = None

        for file in os.listdir(result_dir):
            if file.lower().endswith((".jpg", ".png")):
                output_image_path = os.path.join(result_dir, file)
                break

        if output_image_path:
            st.image(output_image_path, caption="🔍 YOLOv8 Detection Output", use_column_width=True)
        else:
            st.warning("⚠️ Could not generate output image.")
