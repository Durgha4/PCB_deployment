from flask import Flask, render_template, request, url_for
import torch
from PIL import Image
import os
import cv2
import yaml

app = Flask(__name__)

# Load your vehicle detection model
model_path = r'F:\Documents\pcb_detect\pcb_detect\model\best.pt'  # Adjusted path
yaml_path = r'F:\Documents\pcb_detect\pcb_dataset.yaml'     # Path to the YAML file
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Load class names from the YAML file
with open(yaml_path, 'r') as f:
    yaml_data = yaml.safe_load(f)
all_classes = yaml_data.get('names', [])  # Get the 'names' key

# Ensure folders exist
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process based on file type
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Process Image
        processed_file_path, detected_classes = process_image(file_path)
        file_type = 'image'
    elif file.filename.lower().endswith(('.mp4', '.avi')):
        # Process Video
        processed_file_path, detected_classes = process_video(file_path)
        file_type = 'video'
    else:
        return "Unsupported file format", 400

    # Identify missing components
    detected_set = set(detected_classes)
    missing_components = [cls for cls in all_classes if cls not in detected_set]

    # Get URLs for display
    processed_url = url_for('static', filename='processed/' + os.path.basename(processed_file_path))
    upload_url = url_for('static', filename='uploads/' + os.path.basename(file_path))
    return render_template(
        'index.html',
        uploaded_file=upload_url,
        processed_file=processed_url,
        file_type=file_type,
        detected_classes=detected_classes,
        missing_components=missing_components
    )

def process_image(file_path):
    # Load and detect vehicles in the image
    image = Image.open(file_path)
    results = model(image)  # Run detection

    # Extract detected class names
    detected_classes = [model.names[int(cls)] for cls in results.xyxy[0][:, -1]]

    # Render and save processed image
    processed_image_path = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(file_path))
    rendered_image = results.render()[0]
    Image.fromarray(rendered_image).save(processed_image_path)
    return processed_image_path, detected_classes

def process_video(file_path):
    # Open and process video
    cap = cv2.VideoCapture(file_path)
    output_file = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(file_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    detected_classes = set()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on each frame
        results = model(frame)
        detected_classes.update([model.names[int(cls)] for cls in results.xyxy[0][:, -1]])
        processed_frame = results.render()[0]  # Get processed frame
        out.write(processed_frame)  # Write to output video

    cap.release()
    out.release()
    return output_file, list(detected_classes)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)  # Updated for production
