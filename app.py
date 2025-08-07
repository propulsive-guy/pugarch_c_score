from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import base64
from PIL import Image
import io
import os
from datetime import datetime

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("best.pt")  # Ensure this file is present or volume-mounted

@app.route("/", methods=["GET"])
def index():
    return "🚀 YOLOv8 Cleanliness Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"status": "error", "message": "Invalid image"}), 400

    # Run YOLOv8 inference
    results = model(img)[0]

    # Cleanliness Score Calculation
    dirty_classes = [0]  # Update based on your model classes
    total_objects = len(results.boxes)
    dirty_count = 0

    for box in results.boxes:
        cls = int(box.cls.item())
        if cls in dirty_classes:
            dirty_count += 1

    cleanliness_score = 100.0 - ((dirty_count / (total_objects + 1e-5)) * 100.0)
    cleanliness_score = max(0.0, min(100.0, cleanliness_score))  # Clamp 0-100

    # Annotated Image
    annotated = results.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Format response
    response = {
        "status": "success",
        "score": int(cleanliness_score),
        "annotated_image_base64": img_str,
        "metadata": {},
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
