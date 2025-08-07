from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile
import os
import numpy as np
from collections import defaultdict
from PIL import Image
import base64
from datetime import datetime, timezone

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("best.pt")  # Make sure this file exists in the root directory

# Class mappings
class_mapping = {
    0: 'fine dust',
    1: 'garbagebag',
    2: 'liquid',
    3: 'paper_waste',
    4: 'plastic_bottles',
    5: 'plasticbags',
    6: 'stains'
}

# Class weights (for score calculation)
original_weights = {
    0: 1,
    1: 5,
    2: 4,
    3: 2,
    4: 3,
    5: 4,
    6: 3
}
total_weight = sum(original_weights.values())
normalized_weights = {cls: (wt / total_weight * 10) for cls, wt in original_weights.items()}


@app.route("/", methods=["GET"])
def index():
    return "🧼 Cleanliness Score YOLOv8 API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image uploaded"
        }), 400

    file = request.files["image"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        file.save(tmp_file.name)
        image_path = tmp_file.name

    # Run YOLO prediction
    results = model(image_path)[0]
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy().astype(float)  # Ensures JSON serialization

    raw_score = 0
    class_confidence_dict = defaultdict(list)

    # Accumulate weighted scores
    for cls_id, conf in zip(class_ids, confidences):
        weight = normalized_weights.get(cls_id, 1.0)
        raw_score += weight * conf
        class_confidence_dict[cls_id].append(conf)

    cleanliness_score = max(0, round(10.0 - raw_score, 2))
    scaled_score = int(min(100, max(0, cleanliness_score * 10)))  # Final score: 0–100

    # Breakdown by class
    breakdown = []
    for cls_id, conf_list in class_confidence_dict.items():
        avg_conf = float(np.mean(conf_list))
        breakdown.append({
            "class": class_mapping.get(cls_id, str(cls_id)),
            "count": len(conf_list),
            "avg_conf": round(avg_conf, 2),
            "weight": round(normalized_weights[cls_id], 2)
        })

    # Annotated image generation
    encoded_img = None
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
        output_image_path = next(
            (os.path.join(result_dir, f) for f in os.listdir(result_dir)
             if f.lower().endswith((".jpg", ".png"))), None
        )

        if output_image_path:
            with open(output_image_path, "rb") as img_file:
                encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

    # JSON-safe response
    response = {
        "status": "success",
        "score": scaled_score,
        "metadata": {
            "raw_score": round(raw_score, 2),
            "breakdown": breakdown,
            "image_base64": encoded_img
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
