from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import logging
import openai  # OpenAI API integration

try:
    from inference_sdk import InferenceHTTPClient

    USE_ROBOFLOW = True
except ImportError:
    logging.warning("inference-sdk not found! Roboflow API will not be used.")
    USE_ROBOFLOW = False

app = Flask(__name__)

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# OpenAI API Key
openai.api_key = "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"


def get_food_description(food_item):
    """Uses OpenAI to generate a description of the detected food item."""
    prompt = f"Describe the nutritional benefits and uses of {food_item}."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a nutrition expert."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()


def get_meal_recommendations(detected_items):
    """Uses OpenAI to generate meal recommendations based on detected food items."""
    prompt = f"Suggest a meal recipe using {', '.join(detected_items)}."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a chef and nutritionist."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()


@app.route('/nutrition_info', methods=['POST'])
def nutrition_info():
    data = request.json
    if "food_item" not in data:
        return jsonify({"error": "Food item is required"}), 400

    food_item = data["food_item"].lower()
    description = get_food_description(food_item)
    return jsonify({"food_item": food_item, "description": description})


@app.route('/meal_recommendations', methods=['POST'])
def meal_recommendations():
    data = request.json
    if "detected_items" not in data or not isinstance(data["detected_items"], list):
        return jsonify({"error": "A list of detected food items is required"}), 400

    recommendations = get_meal_recommendations(data["detected_items"])
    return jsonify({"detected_items": data["detected_items"], "recommendations": recommendations})

# YOLO Configuration (Local Model)
YOLO_CONFIG = os.path.join(os.getcwd(), "yolov3.cfg")
YOLO_WEIGHTS = os.path.join(os.getcwd(), "yolov3.weights")
YOLO_CLASSES = os.path.join(os.getcwd(), "coco.names")

# Check if YOLO files exist
if not all(os.path.exists(file) for file in [YOLO_CONFIG, YOLO_WEIGHTS, YOLO_CLASSES]):
    logging.error("YOLO model files missing! Ensure yolov3.cfg, yolov3.weights, and coco.names are present.")
    exit(1)

# Load YOLO Model
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
with open(YOLO_CLASSES, "r") as f:
    classes = f.read().strip().split("\n")

# Initialize Roboflow API Client (If available)
if USE_ROBOFLOW:
    ROBOFLOW_CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="rf_0BhRPAg4xwauGp1M0YOVQXaM2hf1"  # Your API Key
    )

# Sample nutrition data per 100g
nutrition_data = {
    "apple": {"calories": 52, "carbs": 14, "protein": 0.3, "fat": 0.2, "fiber": 2.4, "vitamin_c": 4.6, "potassium": 107},
    "banana": {"calories": 89, "carbs": 23, "protein": 1.1, "fat": 0.3, "fiber": 2.6, "vitamin_c": 8.7, "potassium": 358},
    "carrot": {"calories": 41, "carbs": 10, "protein": 0.9, "fat": 0.2, "fiber": 2.8, "vitamin_c": 5.9, "potassium": 320},
    "broccoli": {"calories": 55, "carbs": 11, "protein": 3.7, "fat": 0.6, "fiber": 2.6, "vitamin_c": 89.2, "potassium": 316},
    "potato": {"calories": 77, "carbs": 17, "protein": 2, "fat": 0.1, "fiber": 2.2, "vitamin_c": 19.7, "potassium": 429},
    "orange": {"calories": 47, "carbs": 12, "protein": 0.9, "fat": 0.1, "fiber": 2.4, "vitamin_c": 53.2, "potassium": 181},
    "tomato": {"calories": 18, "carbs": 3.9, "protein": 0.9, "fat": 0.2, "fiber": 1.2, "vitamin_c": 13.7, "potassium": 237},
    "cucumber": {"calories": 15, "carbs": 3.6, "protein": 0.7, "fat": 0.1, "fiber": 0.5, "vitamin_c": 2.8, "potassium": 147},
    "spinach": {"calories": 23, "carbs": 3.6, "protein": 2.9, "fat": 0.4, "fiber": 2.2, "vitamin_c": 28.1, "potassium": 558},
    "capsicum": {"calories": 31, "carbs": 6, "protein": 1, "fat": 0.3, "fiber": 2.1, "vitamin_c": 127.7, "potassium": 211},

}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_images():
    if 'image' not in request.files or 'weight' not in request.form:
        return jsonify({"error": "Image and weight are required"}), 400

    files = request.files.getlist('image')
    weight = request.form.get('weight', type=float)

    if not files or weight <= 0:
        return jsonify({"error": "Invalid image or weight"}), 400

    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    detections_list = []
    nutrition_list = []
    total_nutrition = {"calories": 0, "macros": {"Carbs": 0, "Protein": 0, "Fat": 0, "Fiber": 0}, "micros": {"Vitamin C": 0, "Potassium": 0}}

    for file in files:
        if file.filename == '':
            continue

        file_path = os.path.join(upload_folder, file.filename)
        try:
            file.save(file_path)
            logging.debug(f"File saved at: {file_path}")

            # 1. Try detecting with YOLOv3 (Local)
            detections, nutrition = detect_objects_yolo(file_path, weight)

            # 2. If YOLO fails and Roboflow is available, use it
            if not detections and USE_ROBOFLOW:
                logging.info(f"YOLO failed, switching to Roboflow for: {file.filename}")
                detections, nutrition = detect_objects_roboflow(file_path, weight)

            detections_list.append({"filename": file.filename, "detections": detections})
            nutrition_list.append({"filename": file.filename, "nutrition": nutrition})

            # Sum nutrition values from the current image
            total_nutrition["calories"] += nutrition["total_calories"]
            for macro in nutrition["macros"]:
                total_nutrition["macros"][macro] += nutrition["macros"][macro]
            for micro in nutrition["micros"]:
                total_nutrition["micros"][micro] += nutrition["micros"][micro]

        except Exception as e:
            logging.error(f"Failed to process file {file.filename}: {str(e)}")
            return jsonify({"error": f"Failed to process file {file.filename}: {str(e)}"}), 500

    return jsonify({"detections": detections_list, "nutrition": nutrition_list, "total_nutrition": total_nutrition})


def detect_objects_yolo(image_path, weight):
    """Detect objects using YOLOv3 (Local model)."""
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(output_layers)

    conf_threshold = 0.5
    nms_threshold = 0.4
    boxes, confidences, labels = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                box = detection[:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                labels.append(classes[class_id])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    detected_items = [labels[i] for i in indices.flatten()] if len(indices) > 0 else []
    return detected_items, calculate_nutrition(detected_items, weight)


def detect_objects_roboflow(image_path, weight):
    """Detect objects using Roboflow API (Cloud)."""
    try:
        result = ROBOFLOW_CLIENT.infer(image_path, model_id="vegetables-el4g6/1")
        detected_items = [prediction["class"] for prediction in result["predictions"]]
        return detected_items, calculate_nutrition(detected_items, weight)
    except Exception as e:
        logging.error(f"Roboflow API failed: {str(e)}")
        return [], {"total_calories": 0, "macros": {}, "micros": {}}


def detect_objects_roboflow(image_path, weight):
    """Detect objects using Roboflow API (Cloud)."""
    try:
        result = ROBOFLOW_CLIENT.infer(image_path, model_id="vegetables-el4g6/1")
        print("Roboflow Response:", result)  # Debugging step

        detected_items = [prediction["class"] for prediction in result.get("predictions", [])]
        print("Detected items:", detected_items)  # Debugging step

        if not detected_items:
            logging.warning("No objects detected by Roboflow.")

        return detected_items, calculate_nutrition(detected_items, weight)
    except Exception as e:
        logging.error(f"Roboflow API failed: {str(e)}")
        return [], {"total_calories": 0, "macros": {}, "micros": {}}


def calculate_nutrition(detections, weight):
        """Calculate nutrition values based on detections and weight."""
        total_calories = 0
        macros = {"Carbs": 0, "Protein": 0, "Fat": 0, "Fiber": 0}
        micros = {"Vitamin C": 0, "Potassium": 0}

        for label in detections:
            if label in nutrition_data:
                item_data = nutrition_data[label]
                factor = weight / 100
                total_calories += item_data["calories"] * factor
                for key in macros:
                    macros[key] += item_data.get(key.lower(), 0) * factor
                for key in micros:
                    micros[key] += item_data.get(key.lower(), 0) * factor

        return {"total_calories": round(total_calories, 2), "macros": macros, "micros": micros}



if __name__ == "__main__":
    app.run(debug=True)
