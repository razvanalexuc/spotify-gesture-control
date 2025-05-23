import os
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, request, render_template, session, redirect, url_for
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import base64
import time
from threading import Lock
from flask_sse import sse
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session
app.config['REDIS_URL'] = 'redis://localhost:6379'
app.register_blueprint(sse, url_prefix='/stream')
CORS(app)

# Thread-safe gesture tracking
gesture_lock = Lock()
current_gesture = None
last_gesture_time = 0
GESTURE_COOLDOWN = 2  # seconds

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Spotify API credentials
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
REDIRECT_URI = 'http://127.0.0.1:5000/callback'
TOKEN_URL = 'https://accounts.spotify.com/api/token'
AUTH_URL = f'https://accounts.spotify.com/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={REDIRECT_URI}&scope=user-read-playback-state user-modify-playback-state user-read-currently-playing'

# Global variables
auth_token = None
refresh_token = None

# Gesture detection function
def detect_gesture(pose_landmarks, hand_landmarks):
    global current_gesture, last_gesture_time
    
    current_time = time.time()
    if current_time - last_gesture_time < GESTURE_COOLDOWN:
        return None
    
    # Initialize gesture as None
    detected_gesture = None
    
    if pose_landmarks:
        landmarks = pose_landmarks.landmark
        
        # Calculate distances
        def distance(a, b):
            return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5
        
        # NEXT: Right hand to right shoulder
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # PREVIOUS: Left hand to left shoulder
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        
        # Check for shoulder touches
        if distance(right_hand, right_shoulder) < 0.1:
            detected_gesture = "NEXT"
        elif distance(left_hand, left_shoulder) < 0.1:
            detected_gesture = "PREVIOUS"
    
    # Hand gesture detection
    if hand_landmarks and not detected_gesture:
        # Get hand landmarks
        hand = hand_landmarks.landmark
        
        # Check for thumbs up (Volume Up)
        thumb_tip = hand[4]  # Thumb tip
        index_tip = hand[8]  # Index finger tip
        
        # Thumb is above the index finger (y-coordinate is lower in image space)
        if thumb_tip.y < index_tip.y and thumb_tip.x > hand[0].x:
            detected_gesture = "VOLUME_UP"
        
        # Check for thumbs down (Volume Down)
        elif thumb_tip.y > index_tip.y and thumb_tip.x > hand[0].x:
            detected_gesture = "VOLUME_DOWN"
            
        # Check for open palm (Play/Pause)
        fingers_up = 0
        for fingertip in [8, 12, 16, 20]:  # Tips of index, middle, ring, pinky
            if hand[fingertip].y < hand[fingertip-2].y:  # If tip is above joint
                fingers_up += 1
        
        if fingers_up >= 3:  # Most fingers extended
            detected_gesture = "TOGGLE_PLAY"
    
    # Update global state if a new gesture is detected
    with gesture_lock:
        if detected_gesture and detected_gesture != current_gesture:
            current_gesture = detected_gesture
            last_gesture_time = current_time
            
            # Send SSE update
            sse.publish({"gesture": detected_gesture}, type='gesture')
            
            # Control Spotify
            if auth_token:
                control_spotify(detected_gesture, auth_token)
    
    return detected_gesture

# Spotify API functions
def get_auth_token():
    auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
    
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {"grant_type": "client_credentials"}
    
    response = requests.post(TOKEN_URL, headers=headers, data=data)
    return response.json().get("access_token")

def control_spotify(action, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    base_url = "https://api.spotify.com/v1/me/player"
    
    try:
        if action == "NEXT":
            response = requests.post(f"{base_url}/next", headers=headers)
        elif action == "PREVIOUS":
            response = requests.post(f"{base_url}/previous", headers=headers)
        elif action == "PLAY":
            response = requests.put(f"{base_url}/play", headers=headers)
        elif action == "PAUSE":
            response = requests.put(f"{base_url}/pause", headers=headers)
        elif action == "VOLUME_UP":
            # Get current volume and increase
            current = requests.get(f"{base_url}/volume", headers=headers).json()
            new_vol = min(100, int(current.get('volume_percent', 50)) + 10)
            response = requests.put(f"{base_url}/volume?volume_percent={new_vol}", headers=headers)
        elif action == "VOLUME_DOWN":
            # Get current volume and decrease
            current = requests.get(f"{base_url}/volume", headers=headers).json()
            new_vol = max(0, int(current.get('volume_percent', 50)) - 10)
            response = requests.put(f"{base_url}/volume?volume_percent={new_vol}", headers=headers)
            
        return response.status_code == 204
    except Exception as e:
        print(f"Error controlling Spotify: {e}")
        return False

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose, \
        mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2) as hands:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Convert the BGR image to RGB
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process the image and detect pose and hands
            pose_results = pose.process(image)
            hand_results = hands.process(image)
            
            # Draw the pose and hand annotations on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw pose landmarks
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Draw hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            # Detect gesture
            if pose_results.pose_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0] if hand_results.multi_hand_landmarks else None
                detect_gesture(pose_results.pose_landmarks, hand_landmarks)
            
            # Display current gesture
            with gesture_lock:
                gesture_text = f"Gesture: {current_gesture}" if current_gesture else "No gesture detected"
                cv2.putText(image, gesture_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

# Routes
@app.route('/')
def index():
    # Check if user is authenticated
    token = request.args.get('access_token') or session.get('spotify_token')
    if token:
        session['spotify_token'] = token
        global auth_token
        auth_token = token
    
    return render_template('index.html', auth_url=AUTH_URL)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/auth')
def auth():
    return jsonify({"auth_url": AUTH_URL})

@app.route('/gesture_updates')
def gesture_updates():
    def generate():
        with gesture_lock:
            last_gesture = current_gesture
        
        while True:
            with gesture_lock:
                if current_gesture != last_gesture:
                    last_gesture = current_gesture
                    if current_gesture:
                        yield f"data: {json.dumps({'gesture': current_gesture})}\n\n"
            time.sleep(0.1)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/current_gesture')
def get_current_gesture():
    with gesture_lock:
        return jsonify({"gesture": current_gesture})

@app.route('/callback')
def callback():
    global auth_token, refresh_token
    code = request.args.get('code')
    
    if code:
        # Exchange the code for an access token
        auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
        auth_bytes = auth_string.encode("utf-8")
        auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
        
        headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI
        }
        
        response = requests.post(TOKEN_URL, headers=headers, data=data)
        if response.status_code == 200:
            auth_token = response.json().get("access_token")
            refresh_token = response.json().get("refresh_token")
            return "Successfully authenticated with Spotify! You can close this window."
    
    return "Authentication failed. Please try again."

if __name__ == '__main__':
    # Get initial token if using client credentials flow
    if not auth_token and CLIENT_ID and CLIENT_SECRET:
        auth_token = get_auth_token()
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
