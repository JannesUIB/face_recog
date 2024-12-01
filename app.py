from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import cv2
app = Flask(__name__)

# Load your trained model
model = load_model("face_recognition_model_4.keras")

with open("label_encoder_2.pkl", "rb") as f:
    label_encoder = pickle.load(f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    """
    Preprocess the image by detecting, cropping the face, and resizing it for the model.
    """
    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image_cv, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    
    # Use the first detected face (or handle multiple faces if needed)
    x, y, w, h = faces[0]
    
    # Crop the face region
    face = image_cv[y:y+h, x:x+w]
    
    # Resize the cropped face to the model input size
    face_resized = cv2.resize(face, (160, 160))
    
    # Normalize pixel values
    face_normalized = face_resized / 255.0
    
    # Add batch dimension
    face_batch = np.expand_dims(face_normalized, axis=0)
    plt.imshow(face_batch[0])  # Show the first (and only) image in the batch
    plt.axis("off")  # Turn off axis for better visualization
    plt.show()
    return face_batch

# def preprocess_image(image):
#     """
#     Preprocess the image for the model.
#     - Resize, normalize, etc.
#     """
#     image = image.resize((160, 160))  # Resize to match model input
#     image = np.array(image) / 255.0  # Normalize pixel values
#     image = np.expand_dims(image, axis=0)  # Add batch dimension


#     return image

def decode_base64_image(base64_string):
    """
    Decode a Base64 string to a PIL image.
    """
    base64_data = base64_string.split(",")[1]  # Remove data header
    image_data = BytesIO(base64.b64decode(base64_data))
    return Image.open(image_data)

# Example API endpoint handler
def recognize_face(base64_image):
    """
    Process the Base64 image and recognize the face.
    """
    # Decode Base64 to image
    image = decode_base64_image(base64_image)

    print("the image", image)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    print("preprocessed image?", preprocess_image)
    # Predict using the model
    predictions = model.predict(processed_image)

    print("the predicitions", predictions)
    predicted_class = np.argmax(predictions)  # Get the class ID
    
    print("prediction_class?", predicted_class)

    # Convert index to label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    print("the predicted label", predicted_label)
    # Map class ID to a name or ID
    # person_id = id_mapping.get(predicted_class, "Unknown")
    return ":hehe"

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/capture-face', methods=['POST'])
def capture_face():
    data = request.get_json()
    base64_image = data.get("image")
    
    if not base64_image:
        return jsonify({"error": "No image provided"}), 400
    
    person_id = recognize_face(base64_image)
    return jsonify({"id": person_id})
