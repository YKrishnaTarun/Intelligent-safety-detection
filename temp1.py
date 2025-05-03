import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, send_file, url_for, Response, jsonify, session
import os
import time
from ultralytics import YOLO
import face_recognition
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set your secret key here

# Simulated alert messages storage
alert_messages = []

# Detection categories
safety_violation_objects = ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]
heavy_machinery_objects = ["dump truck", "machinery", "truck", "trailer"]
safety_gear_objects = ["Hardhat", "Mask", "Safety Vest"]

# Store the latest detection message
latest_detection = {"detected": False, "message": "", "objects": [], "timestamp": 0}

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r"C:\Users\karth\skill\New folder\intelligence-safety-firebase-adminsdk-fbsvc-f197e35d3e.json")  # Update this path
firebase_admin.initialize_app(cred)

# Initialize Firestore database
db = firestore.client()

# Initialize Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = r'C:\Users\karth\skill\Object_Detection-main\gen-lang-client-0022288943-c9f40032b8f6.json'  # Update this path

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

def create_drive_folder(user_id):
    try:
        file_metadata = {'name': user_id, 'mimeType': 'application/vnd.google-apps.folder'}
        folder = drive_service.files().create(body=file_metadata, fields='id').execute()
        folder_id = folder.get('id')
        
        # Share folder publicly
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        drive_service.permissions().create(fileId=folder_id, body=permission).execute()
        
        return folder_id
    except Exception as e:
        print(f"Error creating folder in Google Drive: {e}")
        return None

def upload_to_drive(file_path, folder_id):
    try:
        file_metadata = {'name': os.path.basename(file_path), 'parents': [folder_id]}
        media = MediaFileUpload(file_path, resumable=True)
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
        
        # Make file publicly accessible
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        drive_service.permissions().create(fileId=file.get('id'), body=permission).execute()
        
        return file.get('webViewLink')
    except Exception as e:
        print(f"Error uploading to Google Drive: {e}")
        return None

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    user_id = request.form.get("user_id")
    user_ref = db.collection('users').document(user_id)
    user_data = user_ref.get().to_dict()
    folder_id = user_data.get('drive_folder_id') if user_data else None
    
    if not folder_id:
        folder_id = create_drive_folder(user_id)
        if folder_id:
            user_ref.update({'drive_folder_id': folder_id})
        else:
            return jsonify({"error": "Failed to create Google Drive folder."}), 500
    
    drive_link = upload_to_drive(file_path, folder_id)
    if drive_link:
        db.collection("images").add({
            "user_id": user_id,
            "google_drive_url": drive_link,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return jsonify({"message": "File uploaded successfully", "google_drive_url": drive_link})
    else:
        return jsonify({"error": "Failed to upload file to Google Drive."}), 500

def generate_alert_message(detected_objects):
    """Generate message based on detected objects."""
    missing_safety_gear = []
    heavy_machinery = []
    safety_gear_count = 0

    for obj in detected_objects:
        if obj in safety_violation_objects:
            missing_safety_gear.append(obj.replace("NO-", ""))
        elif obj in heavy_machinery_objects:
            heavy_machinery.append(obj)
        elif obj in safety_gear_objects:
            safety_gear_count += 1

    if missing_safety_gear:
        return f"Person without {', '.join(missing_safety_gear)}."
    elif heavy_machinery:
        return f"Heavy machinery detected: {', '.join(heavy_machinery)}."
    elif safety_gear_count >= 3:
        return "The person is with full safety gear."
    else:
        return ""

@app.route('/')
def index1():
    if 'user' in session:
        return redirect(url_for('home1'))
    return render_template('index1.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/admin-login')
def admin_login():
    return render_template('adminlogin.html')

@app.route('/admin-dashboard')
def admin_page():
    return render_template('admindashboard.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route("/report")
def report():
    return render_template("report.html")

@app.route('/login', methods=['POST'])
def login():
    image = request.files['image']
    img = face_recognition.load_image_file(image)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    input_embeddings = face_recognition.face_encodings(img)
    if not input_embeddings:
        return jsonify({'message': 'No face detected. Please try again.'}), 400

    input_embedding = input_embeddings[0]
    users_ref = db.collection('users')
    users = users_ref.stream()

    for user in users:
        user_data = user.to_dict()
        stored_embeddings = user_data.get('face_embeddings')
        if stored_embeddings:
            match = face_recognition.compare_faces([stored_embeddings], input_embedding, tolerance=0.5)
            if match[0]:
                login_time = datetime.datetime.now()
                # Update status to "active" and store login time
                users_ref.document(user.id).update({
                    'status': 'active',
                    'last_login': login_time
                })
                session['user'] = user.id  # Store user in session
                session['login_time'] = time.time()  # Store login time

                # Log login activity
                log_activity(user.id, "logged in")

                # Check if the user has a Google Drive folder
                if not user_data.get('drive_folder_id'):
                    # Create a folder in Google Drive for the user
                    drive_folder_id = create_drive_folder(user.id)
                    if drive_folder_id:
                        # Store the folder ID in Firestore
                        users_ref.document(user.id).update({'drive_folder_id': drive_folder_id})
                    else:
                        return jsonify({'message': 'Failed to create Google Drive folder.'}), 500

                return jsonify({'message': f'Welcome, {user.id}.'}), 200

    return jsonify({'message': 'Unknown user. Please register new user or try again.'}), 400

@app.route('/register', methods=['POST'])
def register():
    try:
        # Get form data
        username = request.form['username'].strip()  # Remove whitespace
        image = request.files['image']
        dob = request.form['dob']  # Get date of birth from form
        
        # Validate inputs
        if not username or not image or not dob:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
            
        # Validate date format (YYYY-MM-DD)
        try:
            datetime.datetime.strptime(dob, '%Y-%m-%d')
        except ValueError:
            return jsonify({'success': False, 'message': 'Invalid date format. Use YYYY-MM-DD'}), 400

        # Process image
        img = face_recognition.load_image_file(image)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        embeddings = face_recognition.face_encodings(img)
        
        if not embeddings:
            return jsonify({'success': False, 'message': 'No face detected. Please try again.'}), 400

        # Check if username already exists
        user_ref = db.collection('users').document(username)
        if user_ref.get().exists:
            return jsonify({'success': False, 'message': 'Username already exists'}), 400

        # Store user data
        registration_time = datetime.datetime.now()
        embeddings = embeddings[0]
        user_ref.set({
            'username': username,
            'registered_at': registration_time,
            'status': 'inactive',
            'face_embeddings': embeddings.tolist(),
            'dob': dob,  # Store the date of birth
            'last_updated': firestore.SERVER_TIMESTAMP
        })
        
        # Log registration activity
        log_activity(username, "registered")
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully!',
            'username': username
        }), 200
        
    except Exception as e:
        print(f"Registration error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Registration failed. Please try again.'
        }), 500
    

@app.route('/traditional-login', methods=['POST'])
def traditional_login():
    try:
        data = request.get_json()
        username = data.get('username')
        dob_input = data.get('dob')

        if not username or not dob_input:
            return jsonify({'success': False, 'message': 'Username and date of birth are required'}), 400

        # Get user from Firestore
        user_ref = db.collection('users').document(username)
        user_data = user_ref.get().to_dict()

        if not user_data:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        # Verify date of birth
        stored_dob = user_data.get('dob')
        if not stored_dob:
            return jsonify({'success': False, 'message': 'Date of birth not registered'}), 401

        # Convert Firestore timestamp to string if necessary
        if hasattr(stored_dob, 'strftime'):  # If it's a datetime object
            stored_dob_str = stored_dob.strftime('%Y-%m-%d')
        else:
            stored_dob_str = stored_dob  # assuming it's already a string

        if dob_input == stored_dob_str:
            # Update status to "active"
            user_ref.update({'status': 'active'})
            session['user'] = username
            session['login_time'] = time.time()
            return jsonify({'success': True, 'message': 'Login successful'}), 200
        else:
            return jsonify({'success': False, 'message': 'Incorrect date of birth'}), 401

    except Exception as e:
        print(f"Traditional login error: {str(e)}")
        return jsonify({'success': False, 'message': 'Login failed. Please try again.'}), 500

@app.route('/logout', methods=['POST'])
def logout():
    if 'user' in session:
        username = session.pop('user', None)
        logout_time = datetime.datetime.now()
        user_ref = db.collection('users').document(username)
        user_ref.update({
            'status': 'inactive',
            'last_logout': logout_time
        })
        
        # Log logout activity
        log_activity(username, "logged out")
        
        return jsonify({'message': f'Goodbye, {username}.'}), 200
    else:
        return jsonify({'message': 'No user logged in.'}), 400

@app.route('/home1')
def home1():
    if 'user' not in session:
        return redirect(url_for('index1'))
    return render_template('home1.html', user=session['user'])

@app.route('/get_navbar_messages')
def get_navbar_messages():
    return jsonify(alert_messages)

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    if "image_path" in request.args:
        image_path = request.args["image_path"]
        return render_template("index.html", image_path=image_path)
    return render_template("index.html")

@app.route("/index", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"error": "No file provided."}), 400

        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        file_extension = f.filename.rsplit('.', 1)[1].lower()

        if file_extension in ['jpg', 'jpeg', 'png']:
            return handle_image(filepath)
        elif file_extension == "mp4":
            return handle_video(filepath)
        else:
            return jsonify({"error": "Unsupported file format."}), 400

    return render_template("index.html", image_path="", media_type='image')

def handle_image(filepath):
    img = cv2.imread(filepath)
    model = YOLO('best.pt')
    detections = model(img, save=True)

    detected_objects = [{"name": model.names[int(detection[5])], "confidence": detection[4].item() * 100} 
                       for detection in detections[0].boxes.data]

    folder_path = os.path.join(os.path.dirname(__file__), 'runs', 'detect')
    latest_subfolder = max([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))], 
                          key=lambda x: os.path.getctime(os.path.join(folder_path, x)))

    detected_files = os.listdir(os.path.join(folder_path, latest_subfolder))
    if not detected_files:
        return jsonify({"error": "No detected files found."}), 404

    detected_file_name = detected_files[0]
    file_path = os.path.join(folder_path, latest_subfolder, detected_file_name)

    time.sleep(2)  # Adjust the delay as needed
    if not os.path.exists(file_path):
        return jsonify({"error": "File not saved by YOLO."}), 500

    user_id = session.get('user')
    if user_id:
        user_ref = db.collection('users').document(user_id)
        user_data = user_ref.get().to_dict()
        drive_folder_id = user_data.get('drive_folder_id')

        if not drive_folder_id:
            return jsonify({"error": "Google Drive folder ID not found for user."}), 500

        image_id = upload_to_drive(file_path, drive_folder_id)
        if image_id:
            db.collection('images').add({
                'user_id': user_id,
                'image_id': image_id,
                'timestamp': datetime.datetime.now()
            })
        else:
            return jsonify({"error": "Failed to upload image to Google Drive."}), 500

    relative_image_path = os.path.relpath(file_path, 'static/assets')
    return render_template('index.html', image_path=relative_image_path, media_type='image', detected_objects=detected_objects)

def handle_video(filepath):
    cap = cv2.VideoCapture(filepath)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (frame_width, frame_height))
    model = YOLO("best.pt")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, save=True)
        res_plotted = results[0].plot()
        cv2.imshow("Result", res_plotted)
        out.write(res_plotted)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    user_id = session.get('user')
    if user_id:
        user_ref = db.collection('users').document(user_id)
        user_data = user_ref.get().to_dict()
        drive_folder_id = user_data.get('drive_folder_id')

        if not drive_folder_id:
            return jsonify({"error": "Google Drive folder ID not found for user."}), 500

        video_id = upload_to_drive("output.mp4", drive_folder_id)
        if video_id:
            db.collection('videos').add({
                'user_id': user_id,
                'video_id': video_id,
                'timestamp': datetime.datetime.now()
            })
        else:
            return jsonify({"error": "Failed to upload video to Google Drive."}), 500

    return render_template('index.html', video_path='output.mp4', media_type='video')
   

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route("/<path:filename>")
def display(filename):
    folder_path = "runs/detect"
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = os.path.join(folder_path, latest_subfolder)
    files = os.listdir(directory)
    latest_file = files[0]
    image_path = os.path.join(directory, latest_file)

    file_extension = latest_file.rsplit(".", 1)[1].lower()
    if file_extension == "jpg":
        return send_file(image_path, mimetype="image/jpeg")
    elif file_extension == "mp4":
        return send_file(image_path, mimetype="video/mp4")
    else:
        return "Invalid file format"

def get_frame():
    video = cv2.VideoCapture("output.mp4")
    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
        )
        time.sleep(0.1)

@app.route("/video_feed")
def video_feed():
    return Response(get_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")
# Store object detection data
detection_reports = []

# Store time spent in the whole day (for example, in seconds)
time_spent_today = 0

# Store person counts for the day
person_counts = []

# Modify your webcam feed to track time and person counts
@app.route("/webcam_feed")
def webcam_feed():
    global latest_detection, time_spent_today
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    # Generate a unique filename for the video
    video_index = 1
    while os.path.exists(f"real_{video_index}.mp4"):
        video_index += 1
    video_filename = f"real_{video_index}.mp4"

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))  # Adjust frame size if needed

    def generate():
        global latest_detection, time_spent_today
        model = YOLO("best.pt")

        while True:
            success, frame = cap.read()
            if not success:
                break

            img = Image.fromarray(frame)
            results = model(img, save=True)

            res_plotted = results[0].plot()
            img_BGR = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)

            # Write the frame to the video file
            out.write(img_BGR)

            ret, buffer = cv2.imencode(".jpg", img_BGR)
            frame = buffer.tobytes()

            # Generate alert message based on detection
            detected_objects = [model.names[int(detection[5])] for detection in results[0].boxes.data]
            message = generate_alert_message(detected_objects)

            if message:
                latest_detection = {
                    "detected": True,
                    "message": message,
                    "objects": detected_objects,
                    "timestamp": time.time()
                }
            else:
                latest_detection = {"detected": False, "message": "", "objects": [], "timestamp": time.time()}
            person_count = detected_objects.count("Person")

            # Store the detection for report purposes
            detection_reports.append({
                'time': time.time() - start_time,  # time since webcam feed started
                'detected_objects': detected_objects,
                'person_count': person_count
            })

            person_counts.append(person_count)

            # Calculate time spent today
            time_spent_today = time.time() - start_time

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
            )

        # Release the VideoWriter and the webcam when done
        out.release()
        cap.release()

        # Upload the video to Google Drive
        user_id = session.get('user')
        if user_id:
            user_ref = db.collection('users').document(user_id)
            user_data = user_ref.get().to_dict()
            drive_folder_id = user_data.get('drive_folder_id')

            if drive_folder_id:
                drive_link = upload_to_drive(video_filename, drive_folder_id)
                if drive_link:
                    db.collection('videos').add({
                        'user_id': user_id,
                        'video_id': drive_link,
                        'timestamp': datetime.datetime.now()
                    })
                    print(f"Video uploaded to Google Drive: {drive_link}")
                else:
                    print("Failed to upload video to Google Drive.")
            else:
                print("Google Drive folder ID not found for user.")
        else:
            print("User not logged in.")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_latest_detection")
def get_latest_detection():
    current_time = time.time()
    detected_objects = latest_detection["objects"]
    person_count = detected_objects.count("Person")  # Count occurrences of 'person' in detected objects

    if latest_detection["detected"] and (current_time - latest_detection["timestamp"] < 5):
        # Include person_count in the response
        return jsonify({
            "detected": True,
            "message": latest_detection["message"],
            "objects": detected_objects,
            "person_count": person_count  # Send person count as part of the response
        })
    else:
        return jsonify({"detected": False, "message": "", "objects": [], "person_count": 0})  # Send 0 if no detection

camera_active = False  # Camera is OFF by default

def seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

@app.route('/get_report_data')
def get_report_data():
    if 'user' not in session:
        return jsonify({"error": "User not logged in"}), 401
    
    # Get the login time from the session
    login_time = session.get('login_time', time.time())  # Default to current time if not set
    
    # Calculate time spent since login
    current_time = time.time()
    time_spent_seconds = current_time - login_time
    
    # Convert time spent to HH:MM:SS format
    time_spent_formatted = seconds_to_hms(time_spent_seconds)
    
    # Calculate total person count for the day
    total_person_count = sum(person_counts)
    
    # Get the latest person count (the most recent value from the person_counts list)
    latest_person_count = person_counts[-1] if person_counts else 0  # Default to 0 if no data
    
    # Prepare data for the chart (time spent, person count, etc.)
    return jsonify({
        "time_spent_today": time_spent_formatted,  # Return formatted time
        "person_counts": person_counts,
        "detection_reports": detection_reports,
        "total_person_count": total_person_count,  # Add the total count of persons
        "latest_person_count": latest_person_count  # Add the latest person count (real-time)
    })

@app.route('/turn_on_camera', methods=['POST'])
def turn_on_camera():
    global camera_active
    camera_active = True  
    return jsonify({"status": "on"})

@app.route('/turn_off_camera', methods=['POST'])
def turn_off_camera():
    global camera_active
    camera_active = False  
    return jsonify({"status": "off"})

@app.route('/get_latest_alerts')
def get_latest_alerts():
    return jsonify({"messages": alert_messages})
# Function to log user activity
def log_activity(username, action):
    logs_ref = db.collection('logs')
    logs_ref.add({
        'username': username,
        'action': action,
        'timestamp': datetime.datetime.now()
    })

# Function to fetch user statistics from Firestore
def get_user_stats():
    users_ref = db.collection('users')
    users = users_ref.stream()
    
    total_users = 0
    active_users = 0
    pending_registrations = 0
    inactive_users = 0
    
    for user in users:
        total_users += 1
        user_data = user.to_dict()
        
        # Check if the user has completed registration
        if 'face_embeddings' not in user_data:
            pending_registrations += 1
        elif user_data.get('status') == 'active':
            active_users += 1
        else:
            inactive_users += 1
    
    return {
        'total_users': total_users,
        'active_users': active_users,
        'pending_registrations': pending_registrations,
        'inactive_users': inactive_users
    }

# Function to fetch recent activity logs from Firestore
def get_recent_activity():
    logs_ref = db.collection('user_activity_logs')
    
    # Fetch the last 4 recent logs, ordered by timestamp in descending order
    logs = logs_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(4).stream()
    
    recent_activity = []
    for log in logs:
        log_data = log.to_dict()
        # Format the timestamp to a more readable format
        formatted_timestamp = log_data.get('timestamp').strftime("%Y-%m-%d %H:%M:%S")
        
        recent_activity.append({
            'username': log_data.get('username'),
            'action': log_data.get('action'),
            'timestamp': formatted_timestamp,
            
        })
    
    return recent_activity

def log_activity(username, action):
    logs_ref = db.collection('user_activity_logs')
    logs_ref.add({
        'username': username,
        'action': action,
        'timestamp': datetime.datetime.now(),
        
    })

# Route to render the admin dashboard
@app.route('/admin-dashboard')
def admin_dashboard():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    
    # Fetch user statistics
    user_stats = get_user_stats()
    
    # Fetch recent activity logs
    recent_activity = get_recent_activity()
    
    return render_template('admindashboard.html', 
                           total_users=user_stats['total_users'],
                           active_users=user_stats['active_users'],
                           pending_registrations=user_stats['pending_registrations'],
                           inactive_users=user_stats['inactive_users'],
                           recent_activity=recent_activity)

# Route to fetch user statistics (for AJAX updates)
@app.route('/get_user_stats')
def fetch_user_stats():
    user_stats = get_user_stats()
    return jsonify(user_stats)

# Route to fetch recent activity logs (for AJAX updates)
@app.route('/get_recent_activity')
def fetch_recent_activity():
    recent_activity = get_recent_activity()
    return jsonify(recent_activity)
# Function to add new alert messages (use real detection logic here)
def add_alert_message(text):
    formatted_text = f"Hello Inspector, there is a person with {text}"
    alert_messages.append({
        "text": formatted_text,
        "timestamp": datetime.datetime.now().strftime("%I:%M %p")  # Format: "2:30 PM"
    })
    if len(alert_messages) > 5:  # Keep only latest 5 messages
        alert_messages.pop(0)
@app.route('/get_all_users')
def get_all_users():
    users_ref = db.collection('users')
    users = users_ref.stream()
    user_list = []
    for user in users:
        user_data = user.to_dict()
        # Format the registered date to a readable format
        registered_date = user_data.get('registered_at', 'N/A')
        if registered_date != 'N/A':
            registered_date = registered_date.strftime("%Y-%m-%d %H:%M:%S")
        user_list.append({
            'name': user_data.get('username', 'N/A'),
            'status': user_data.get('status', 'N/A'),
            'registered_date': registered_date  # Add registered date
        })
    return jsonify(user_list)

@app.route('/get_active_users')
def get_active_users():
    users_ref = db.collection('users').where('status', '==', 'active')
    users = users_ref.stream()
    user_list = []
    for user in users:
        user_data = user.to_dict()
        user_list.append({
            'name': user_data.get('username', 'N/A'),
            'status': user_data.get('status', 'N/A')
        })
    return jsonify(user_list)

@app.route('/get_pending_users')
def get_pending_users():
    users_ref = db.collection('users').where('status', '==', 'pending')
    users = users_ref.stream()
    user_list = []
    for user in users:
        user_data = user.to_dict()
        user_list.append({
            'name': user_data.get('username', 'N/A'),
            'status': user_data.get('status', 'N/A')
        })
    return jsonify(user_list)

@app.route('/get_inactive_users')
def get_inactive_users():
    users_ref = db.collection('users').where('status', '==', 'inactive')
    users = users_ref.stream()
    user_list = []
    for user in users:
        user_data = user.to_dict()
        user_list.append({
            'name': user_data.get('username', 'N/A'),
            'status': user_data.get('status', 'N/A')
        })
    return jsonify(user_list)

@app.route('/remove_user', methods=['POST'])
def remove_user():
    username = request.form.get('username')
    if not username:
        return jsonify({'success': False, 'message': 'Username is required.'}), 400

    try:
        # Delete the user from Firestore
        user_ref = db.collection('users').document(username)
        user_ref.delete()
        return jsonify({'success': True, 'message': f'User {username} removed successfully.'}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/get_drive_folder_link', methods=['GET'])
def get_drive_folder_link():
    user_id = session.get('user')
    if not user_id:
        return jsonify({"error": "User not logged in."}), 401

    user_ref = db.collection('users').document(user_id)
    user_data = user_ref.get().to_dict()
    if not user_data:
        return jsonify({"error": "User data not found."}), 404

    drive_folder_id = user_data.get('drive_folder_id')
    if not drive_folder_id:
        return jsonify({"error": "Google Drive folder ID not found."}), 404

    # Construct the Google Drive folder link
    drive_folder_link = f"https://drive.google.com/drive/folders/{drive_folder_id}"
    return jsonify({"drive_folder_link": drive_folder_link})

# Simulating new alerts (replace with actual detection logic)
import threading, time
def simulate_alerts():
    while True:
        add_alert_message(" no Safety gear detected!")
        time.sleep(5)  # Every 10 seconds

# Run background simulation
threading.Thread(target=simulate_alerts, daemon=True).start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = YOLO("best.pt")
    app.run(host="0.0.0.0", port=args.port, debug=True)