from flask import Flask, render_template, request, jsonify,redirect, url_for, session
import base64
import psycopg2
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import cv2
import math
from datetime import datetime, timedelta, time
from flask_bcrypt import Bcrypt
import bcrypt
from collections import Counter

app = Flask(__name__)
app.secret_key = 'uib_ai_recog'  # Use a strong secret key
# bcrypt = Bcrypt(app)
def get_db_connection():
    conn = psycopg2.connect(host='localhost',
                            database='ai_recog_db',
                            user="postgres",
                            password="postgres")
    return conn

UIB_LAT = 1.1208372285337016
UIB_LONG = 104.00328410332459

waktu_masuk = time(8,0)
waktu_pulang = time(17,0)
# Load your trained model
model = load_model("face_recognition_model_4.keras")

with open("label_encoder_2.pkl", "rb") as f:
    label_encoder = pickle.load(f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth using the Haversine formula.
    """
    # Radius of the Earth in meters
    R = 6371000

    # Convert latitude and longitude from degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in meters
    distance = R * c
    return distance

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

    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Predict using the model
    predictions = model.predict(processed_image)

    predicted_class = np.argmax(predictions)  # Get the class ID
    

    # Convert index to label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label


# Route to render login page
@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

# Route to handle login
@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    passwords = request.form['password']

    # Connect to the database
    conn = get_db_connection()
    cur = conn.cursor()

    # Query the user
    cur.execute("SELECT * FROM admin WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()

    print("password", passwords.encode('utf-8'))
    print("password", user[2].encode('utf-8'))
    # Validate user credentials
    if not user or not bcrypt.checkpw(passwords.encode('utf-8'), user[2].encode('utf-8')):
        return jsonify({'error': 'Invalid Email or password'}), 401

    # Save user session
    session['user_id'] = user[0]
    session['username'] = user[1]
    session['email'] = user[3]
    return redirect(url_for('index'))

# Logout route
@app.route('/logout', methods=['GET'])
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route("/get_image_by_karyawan_id" , methods=['POST'])
def get_image_karyawan():
    data = request.get_json()
    karyawan_id = data.get("id")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM karyawan where npm = %s;", (karyawan_id,))
    karyawan = cur.fetchall()
    cur.close()
    conn.close()

    image_base64 = base64.b64encode(karyawan[0][2].tobytes()).decode('utf-8')
    
    return jsonify({"image_base64" : image_base64})


@app.route("/take_attendance" , methods=['POST'])
def post_take_attendance():
    data = request.get_json()
    karyawan_id = data.get("id")
    attendance_status = data.get("status")

    now = datetime.now()

    if attendance_status == "" or False:
        return jsonify({'error' : "Attedance Status Is Invalid, Please Check With IT Team"})
    conn = get_db_connection()
    cur = conn.cursor()

    if attendance_status == "Not Attended":
        cur.execute("UPDATE karyawan set status = 'Attended' where npm = %s;", (karyawan_id,))
        cur.execute("INSERT INTO absensi (karyawan_id, waktu_masuk, status) VALUES (%s,%s,%s);", (karyawan_id,now,'Masuk'))

    elif attendance_status == "Attended":
        cur.execute("UPDATE karyawan set status = 'Not Attended' where npm = %s;", (karyawan_id,))

        if now.time() >= waktu_masuk and now.time() <= waktu_pulang:
            cur.execute("INSERT INTO absensi (karyawan_id, waktu_keluar, status) VALUES (%s,%s,%s);", (karyawan_id,now,'Keluar'))
        else:
            cur.execute("INSERT INTO absensi (karyawan_id, waktu_keluar, status) VALUES (%s,%s,%s);", (karyawan_id,now,'Pulang'))
    
    conn.commit()
    cur.close()
    conn.close()

    # karyawan = cur.fetchall()
    # image_base64 = base64.b64encode(karyawan[0][2].tobytes()).decode('utf-8')
    
    return jsonify({"sucesss" : True})

@app.route('/')
def index():
    is_token = request.args.get("token") or False
    is_admin = False
    if session.get('user_id'):
        is_admin = True

    return render_template('index.html', is_admin=is_admin, is_token=is_token)

@app.route('/report')
def report():
    is_admin = False
    if session.get('user_id'):
        is_admin = True

    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    end_time = datetime.combine(now.date(), datetime.max.time())

    # Get the search query from the request
    search_query = request.args.get('search', '')

    conn = get_db_connection()
    cur = conn.cursor()
    if search_query:
        cur.execute("""
            SELECT absensi.*, karyawan.nama FROM absensi
            JOIN karyawan ON absensi.karyawan_id = karyawan.npm
            WHERE (absensi.karyawan_id::text LIKE %s OR karyawan.nama LIKE %s)
            AND (waktu_masuk >= %s OR waktu_keluar <= %s);
        """, ('%' + search_query + '%', '%' + search_query + '%', start_time, end_time))
    else:
        cur.execute("""
            SELECT absensi.*, karyawan.nama FROM absensi
            JOIN karyawan ON absensi.karyawan_id = karyawan.npm
            WHERE (waktu_masuk >= %s OR waktu_keluar <= %s);
        """, (start_time, end_time))
    absen_ids = cur.fetchall()
    cur.close()
    conn.close()
    
    statuses = [row[4].strip() for row in absen_ids]
    status_counts = Counter(statuses)
    return render_template('report.html', is_admin=is_admin, masuk_count=status_counts.get('Masuk', 0), keluar_count=status_counts.get('Keluar', 0),pulang_count=status_counts.get('Pulang', 0), data_absen = absen_ids)

@app.route('/confirmation')
def confirmation():
    id = request.args.get("id")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM karyawan where npm = %s;", (id,))
    karyawan_id = cur.fetchall()
    cur.close()
    conn.close()

    is_admin = False
    if session.get('user_id'):
        is_admin = True
    return render_template("absen_confirmation.html", id=id, name=karyawan_id[0][4], email=karyawan_id[0][3], status=karyawan_id[0][1], is_admin=is_admin)



@app.route('/api/capture-face', methods=['POST'])
def capture_face():
    data = request.get_json()
    base64_image = data.get("image")
    lat = data.get("latitude")
    long = data.get("longitude")

    distance = haversine_distance(UIB_LAT, UIB_LONG, lat, long)

    # Check if within 300 meters
    if distance <= 300:
        print(f"Within 300 meters. Distance: {distance:.2f} meters")
    
        if not base64_image:
            return jsonify({"error": "No image provided"}), 400
        
        person_id = recognize_face(base64_image)
        return jsonify({"success" : True,"id": person_id})
    else:
        return jsonify({"error": "You Are Too Far"}), 400


@app.route('/api/token_attendance', methods=['POST'])
def token_attendance():
    data = request.get_json()
    token = data.get("token_input")
    lat = data.get("latitude")
    long = data.get("longitude")

    distance = haversine_distance(UIB_LAT, UIB_LONG, lat, long)

    # Check if within 300 meters
    # if distance <= 300:
    print(f"Within 300 meters. Distance: {distance:.2f} meters")

    if not token:
        return jsonify({"error": "No Token provided"}), 400
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM karyawan where token = %s;", (token,))
    karyawan_id = cur.fetchall()
    cur.close()
    conn.close()

    if not karyawan_id:
        return jsonify({"error": "No Data Found"}), 400
    return jsonify({"success" : True,"id": karyawan_id[0][0]})
    # else:
    #     return jsonify({"error": "You Are Too Far"}), 400
