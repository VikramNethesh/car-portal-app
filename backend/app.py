# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Placeholder: load ML models here
# yolo_model = load_yolo_model()
# tire_net = load_tire_net()

@app.route("/")
def index():
    return jsonify({"message": "Car Data Extraction API running!"})

@app.route("/process-car", methods=["POST"])
def process_car():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save uploaded file temporarily
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    # --- ML Pipeline (pseudo-code for now) ---
    # car_info = yolo_model.detect_car(filepath)
    # plate_info = yolo_model.detect_plate(filepath)
    # dents = yolo_model.detect_dents(filepath)
    # tire_info = tire_net.analyze(filepath)

    # For now, dummy data
    result = {
        "company": "Toyota",
        "model": "Corolla",
        "plate_number": "TN-09-AB-1234",
        "dents": ["front_left", "rear_bumper"],
        "tire_condition": "minor deformation on front-right tire"
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
