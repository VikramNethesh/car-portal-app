from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
from io import BytesIO

# --- Configuration ---
app = Flask(__name__)
CORS(app)

# Use a safe temporary directory for uploads and crops
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "tmp")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --- Global Model and Resource Initialization ---
# Define global variables for models and data
car_detector = None
plate_detector = None
logo_detector = None
logo_classifier = None
ocr_tokenizer = None
ocr_model = None
logo_classes = []

# --- Utility Functions ---

def detect_car_color(image_path, k=3):
    """
    Analyzes the image using K-Means clustering to find the dominant color.
    This uses the simplified logic from the notebook.
    """
    img = cv2.imread(image_path)
    if img is None:
        return "Unknown (Color Load Error)"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img, (100, 100))
    img_data = img_small.reshape((-1, 3))

    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(img_data)
        dominant_color = kmeans.cluster_centers_.astype(int)[0]
        r, g, b = dominant_color.tolist()

        # Simple mapping to human-readable color
        if r > 200 and g > 200 and b > 200:
            return "White"
        elif r < 50 and g < 50 and b < 50:
            return "Black"
        elif r > g and r > b:
            return "Red"
        elif g > r and g > b:
            return "Green"
        elif b > r and b > g:
            return "Blue"
        else:
            return f"RGB({r}, {g}, {b})"
    except Exception as e:
        return f"Color Analysis Error: {e}"


def load_all_models():
    """Load all machine learning models once at startup."""
    global car_detector, plate_detector, logo_detector, logo_classifier
    global ocr_tokenizer, ocr_model, logo_classes

    # Ensure model directory exists (e.g., if downloading weights)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("--- Starting Model Load ---")

    try:
        # 1. OCR Model (from Hugging Face)
        print("Loading OCR model...")
        ocr_tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        ocr_model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        ocr_model.eval().cpu() # Use .cpu() or .cuda() as appropriate for your environment

        # 2. YOLO Models (Assume models are placed in the MODEL_DIR for production)
        print("Loading YOLO models...")
        # Note: In a real deployment, you'd download these models and specify their paths.
        # For this example, we'll use placeholder paths based on the notebook structure:
        car_detector = YOLO(os.path.join(MODEL_DIR, "yolo12s.pt"))
        plate_detector = YOLO(os.path.join(MODEL_DIR, "yolo12lic-pytorch-default-v1/runs/runs/detect/train2/weights/best.pt"))
        logo_detector = YOLO(os.path.join(MODEL_DIR, "car_logo_detector-pytorch-default-v1/detect/train/weights/best.pt"))
        logo_classifier = YOLO(os.path.join(MODEL_DIR, "yolo-logocls-pytorch-default-v1/yolo_classification_model.pt"))
        
        # 3. Load Logo Class Names
        print("Loading class names...")
        with open(os.path.join(MODEL_DIR, "yolo-logocls-pytorch-default-v1/class_names.txt"), "r") as f:
             logo_classes = [line.strip() for line in f.readlines()]
             
        print("--- All Models Loaded Successfully ---")

    except Exception as e:
        print(f"FATAL ERROR: Could not load all models. Check paths in MODEL_DIR. Error: {e}")
        # In a production environment, you would exit or raise an exception here.


# --- Core Logic Function ---

def run_car_analysis_pipeline(filepath, temp_car_dir):
    """
    Executes the multi-step car analysis pipeline on a given image file.
    """
    if not all([car_detector, plate_detector, logo_detector, logo_classifier, ocr_model, ocr_tokenizer]):
        return {"error": "ML models failed to initialize."}
        
    result = {
        "license_number": "Not Detected",
        "car_brand": "Unknown",
        "dominant_color": "Unknown"
    }

    try:
        image = Image.open(filepath).convert("RGB")
        car_results = car_detector(np.array(image), verbose=False)
        car_boxes = car_results[0].boxes.xyxy.cpu().numpy()
    except Exception as e:
        return {"error": f"Initial detection failed: {e}"}

    if len(car_boxes) == 0:
        return {"error": "No car detected in the image."}
        
    # Process the first detected car (index 0)
    car_box = car_boxes[0]
    x1, y1, x2, y2 = map(int, car_box)
    car_crop = image.crop((x1, y1, x2, y2))
    
    # Save car crop for color detection
    car_crop_path = os.path.join(temp_car_dir, "car_crop.jpg")
    car_crop.save(car_crop_path)

    # 1. Detect Plate & OCR
    plate_results = plate_detector(np.array(car_crop), verbose=False)
    plate_boxes = plate_results[0].boxes.xyxy.cpu().numpy()

    ##print("Starting ocr")
    ##if len(plate_boxes) > 0:
    ##    pbox = plate_boxes[0]
    ##    px1, py1, px2, py2 = map(int, pbox)
    ##    plate_crop = car_crop.crop((px1, py1, px2, py2))
    ##    plate_path = os.path.join(temp_car_dir, "plate.jpg")
    ##    plate_crop.save(plate_path)

    ##    res = ocr_model.chat(ocr_tokenizer, plate_path, ocr_type="ocr")
    ##    if isinstance(res, str):
    ##        result["license_number"] = res.strip()
    ##    elif isinstance(res, dict) and "text" in res:
    ##        result["license_number"] = res["text"].strip() 

    # 2. Car Logo Detection and Classification
    logo_results = logo_detector.predict(np.array(car_crop), conf=0.3, verbose=False)
    
    for r in logo_results:
        orig_img = cv2.cvtColor(np.array(car_crop), cv2.COLOR_RGB2BGR)
        boxes = r.boxes.xyxy.cpu().numpy()
        
        if len(boxes) > 0:
            box = boxes[0] # Focus on the best logo detection
            x1_logo, y1_logo, x2_logo, y2_logo = map(int, box)
            logo_crop = orig_img[y1_logo:y2_logo, x1_logo:x2_logo]

            # 4. Logo Classification
            cls_result = logo_classifier.predict(logo_crop, verbose=False)
            cls_id = int(cls_result[0].probs.top1)
            cls_name = logo_classes[cls_id]
            # cls_conf = float(cls_result[0].probs.top1conf) # Can be used for confidence check
            
            result["car_brand"] = cls_name
            break # Exit after successfully classifying one logo

    # 5. Dominant Color Extraction
    result["dominant_color"] = detect_car_color(car_crop_path)
    
    return result


# --- Flask Routes ---

@app.route("/")
def index():
    return jsonify({"message": "Car Data Extraction API running!"})

@app.route("/process-car", methods=["POST"])
def process_car():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # 1. Setup temporary file paths
    temp_dir_id = os.urandom(8).hex()
    temp_car_dir = os.path.join(UPLOAD_DIR, temp_dir_id)
    os.makedirs(temp_car_dir, exist_ok=True)
    
    filepath = os.path.join(temp_car_dir, file.filename)
    file.save(filepath)

    try:
        # 2. Run the analysis pipeline
        analysis_result = run_car_analysis_pipeline(filepath, temp_car_dir)
        
        if "error" in analysis_result:
             return jsonify(analysis_result), 500
        
        # Structure the final response
        final_result = {
            "company": analysis_result["car_brand"].split(".")[0], # Simple extraction of company
            "model": analysis_result["car_brand"].split(".")[-1],  # Simple extraction of model/common
            "license_number": analysis_result["license_number"],
            "dominant_color": analysis_result["dominant_color"],
            # Placeholder for other data (dents, tires) if you add models later
            "dents": ["analysis pending"],
            "tire_condition": "analysis pending"
        }
        
        return jsonify(final_result)
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return jsonify({"error": f"An unexpected error occurred during processing: {e}"}), 500
        
    finally:
        # 3. Cleanup: Remove the temporary directory and all its contents
        if os.path.exists(temp_car_dir):
            import shutil
            shutil.rmtree(temp_car_dir)


if __name__ == "__main__":
    # Load models before running the app
    load_all_models()
    
    # Ensure the main upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # The debug setting here is generally bad practice for production but fine for development.
    app.run(host='0.0.0.0', port=5000, debug=True)