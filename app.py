from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile
import os
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone

app = Flask(__name__)

# Load model
model = YOLO("best.pt")  # Ensure best.pt matches the updated 10-class model

# Updated class mapping (index → label)
class_mapping = {
    0: 'clean_indian',
    1: 'clean_urinal',
    2: 'clean_western',
    3: 'damage',
    4: 'dirty_basin',
    5: 'dirty_floor',
    6: 'dirty_indian',
    7: 'dirty_urinal',
    8: 'dirty_western',
    9: 'garbage'
}

# Class importance weights (you can adjust these if some classes are more critical)
original_weights = {
    0: 0.5,  # clean_indian → positive class, low impact
    1: 0.5,  # clean_urinal
    2: 0.5,  # clean_western
    3: 4,    # damage
    4: 3,    # dirty_basin
    5: 4,    # dirty_floor
    6: 4,    # dirty_indian
    7: 3,    # dirty_urinal
    8: 4,    # dirty_western
    9: 5     # garbage → most severe
}

# Normalize weights to a 0–10 scale
total_weight = sum(original_weights.values())
normalized_weights = {cls: (wt / total_weight * 10) for cls, wt in original_weights.items()}


@app.route("/", methods=["GET"])
def home():
    return "🚽 Restroom Cleanliness YOLOv8 API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    if 'images' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No images uploaded. Please upload using the 'images' field (multipart/form-data)."
        }), 400

    files = request.files.getlist('images')
    responses = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            file.save(tmp_file.name)
            image_path = tmp_file.name

        try:
            # Run YOLO inference
            results = model(image_path)[0]
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy().astype(float)

            raw_score = 0
            class_confidence_dict = defaultdict(list)

            # Weighted cleanliness scoring
            for cls_id, conf in zip(class_ids, confidences):
                weight = normalized_weights.get(cls_id, 1.0)
                raw_score += weight * conf
                class_confidence_dict[cls_id].append(conf)

            # Invert scale so higher score = cleaner (0–10)
            cleanliness_score = max(0, round(10.0 - raw_score, 2))

            # Breakdown per class
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
