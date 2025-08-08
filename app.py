from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile
import os
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone

app = Flask(__name__)

# Load model
model = YOLO("best.pt")  # Make sure best.pt is in the working directory

# Class mapping (index to label)
class_mapping = {
    0: 'fine dust',
    1: 'garbagebag',
    2: 'liquid',
    3: 'paper_waste',
    4: 'plastic_bottles',
    5: 'plasticbags',
    6: 'stains'
}

# Class importance weights
original_weights = {
    0: 1,
    1: 5,
    2: 4,
    3: 2,
    4: 3,
    5: 4,
    6: 3
}

# Normalize weights to a base-10 scale
total_weight = sum(original_weights.values())
normalized_weights = {cls: (wt / total_weight * 10) for cls, wt in original_weights.items()}


@app.route("/", methods=["GET"])
def home():
    return "🧼 Cleanliness Score YOLOv8 API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    if 'images' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No images uploaded. Please upload using 'images' field as multipart/form-data."
        }), 400

    files = request.files.getlist('images')
    responses = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            file.save(tmp_file.name)
            image_path = tmp_file.name

        try:
            results = model(image_path)[0]
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy().astype(float)

            raw_score = 0
            class_confidence_dict = defaultdict(list)

            for cls_id, conf in zip(class_ids, confidences):
                weight = normalized_weights.get(cls_id, 1.0)
                raw_score += weight * conf
                class_confidence_dict[cls_id].append(conf)

            cleanliness_score = max(0, round(10.0 - raw_score, 2))  # Clamp to 0–10

            breakdown = []
            for cls_id, conf_list in class_confidence_dict.items():
                avg_conf = float(np.mean(conf_list))
                breakdown.append({
                    "class": class_mapping.get(cls_id, str(cls_id)),
                    "count": len(conf_list),
                    "avg_conf": round(avg_conf, 2),
                    "weight": round(normalized_weights.get(cls_id, 1.0), 2)
                })

            responses.append({
                "status": "success",
                "score": cleanliness_score,
                "metadata": {
                    "raw_score": round(raw_score, 2),
                    "breakdown": breakdown
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "filename": file.filename
            })

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

    return jsonify(responses)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
