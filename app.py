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
mp_face_mesh = mp.solutions.face_mesh
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
def detect_gesture(pose_landmarks, hand_landmarks, face_landmarks=None):
    global current_gesture, last_gesture_time
    
    current_time = time.time()
    if current_time - last_gesture_time < GESTURE_COOLDOWN:
        return current_gesture  # Return current gesture if still in cooldown
    
    # Initialize gesture as None
    detected_gesture = None
    
    try:
        if pose_landmarks and hand_landmarks:
            landmarks = pose_landmarks.landmark
            
            # Calculate normalized distance between two points
            def distance(a, b):
                return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5
            
            # Get required pose landmarks
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # Get hand landmarks
            hand_landmarks = hand_landmarks.landmark
            wrist = hand_landmarks[mp_hands.HandLandmark.WRIST.value]
            
            # 1. Check for hand-to-shoulder gestures (more reliable)
            SHOULDER_THRESHOLD = 0.15  # Distance threshold for shoulder touch
            
            # Calculate distances from hand to shoulders
            left_shoulder_dist = distance(wrist, left_shoulder)
            right_shoulder_dist = distance(wrist, right_shoulder)
            
            # Check which shoulder is closer to the hand
            if left_shoulder_dist < right_shoulder_dist and left_shoulder_dist < SHOULDER_THRESHOLD:
                detected_gesture = "NEXT_TRACK"
                print(f"Right shoulder touch detected ({left_shoulder_dist:.3f}) -> Next track")
            elif right_shoulder_dist < SHOULDER_THRESHOLD:
                detected_gesture = "PREVIOUS_TRACK"
                print(f"Left shoulder touch detected ({right_shoulder_dist:.3f}) -> Previous track")
            
            # 2. Check for cheek touches (if no shoulder touch detected and face landmarks are available)
            if not detected_gesture and face_landmarks and hand_landmarks and hasattr(hand_landmarks, 'landmark') and len(hand_landmarks.landmark) > 8:
                try:
                    # Get hand wrist position
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST.value]
                    
                    # Get cheek landmarks (indices for left and right cheek points in face mesh)
                    # Note: These indices might need adjustment based on the exact face mesh model
                    LEFT_CHEEK_INDEX = 234  # Approximate index for left cheek
                    RIGHT_CHEEK_INDEX = 454  # Approximate index for right cheek
                    
                    # Access face mesh landmarks correctly
                    if hasattr(face_landmarks, 'landmark'):
                        left_cheek = face_landmarks.landmark[LEFT_CHEEK_INDEX]
                        right_cheek = face_landmarks.landmark[RIGHT_CHEEK_INDEX]
                    else:
                        # If face_landmarks is already the landmark list
                        left_cheek = face_landmarks[LEFT_CHEEK_INDEX]
                        right_cheek = face_landmarks[RIGHT_CHEEK_INDEX]
                    
                    # Calculate distances from hand to cheeks
                    left_cheek_dist = distance(wrist, left_cheek)
                    right_cheek_dist = distance(wrist, right_cheek)
                    
                    # Check for cheek touches
                    CHEEK_THRESHOLD = 0.1  # Adjust this value for sensitivity
                    
                    if left_cheek_dist < CHEEK_THRESHOLD:
                        detected_gesture = "VOLUME_DOWN"
                        print(f"Left cheek touch detected ({left_cheek_dist:.3f}) -> Volume Down")
                    elif right_cheek_dist < CHEEK_THRESHOLD:
                        detected_gesture = "VOLUME_UP"
                        print(f"Right cheek touch detected ({right_cheek_dist:.3f}) -> Volume Up")
                except (IndexError, AttributeError) as e:
                    # Skip if there's an error accessing landmarks
                    print(f"Error accessing face landmarks: {e}")
                    pass
            
            # 3. Thumb detection (if no cheek or shoulder touch detected)
            if not detected_gesture and len(hand_landmarks) > 8:
                thumb_tip = hand_landmarks[4]  # Thumb tip
                index_tip = hand_landmarks[8]   # Index finger tip
                
                # Calculate distances for thumb detection
                thumb_index_dist = distance(thumb_tip, index_tip)
                thumb_wrist_dist = distance(thumb_tip, wrist)
                
                # Thumb up/down detection
                THUMB_THRESHOLD = 0.08  # Adjust this based on testing
                
                # For thumb up/down, we'll check the y-position relative to the wrist
                if thumb_tip.y < wrist.y - 0.1:  # Thumb is above wrist
                    if thumb_index_dist < THUMB_THRESHOLD:
                        detected_gesture = "VOLUME_UP"
                        print("Thumbs up detected")
                elif thumb_tip.y > wrist.y + 0.1:  # Thumb is below wrist
                    if thumb_index_dist < THUMB_THRESHOLD:
                        detected_gesture = "VOLUME_DOWN"
                        print("Thumbs down detected")
            
            # 3. Update gesture state if a new gesture is detected
            if detected_gesture and detected_gesture != current_gesture:
                with gesture_lock:
                    current_gesture = detected_gesture
                last_gesture_time = current_time
                
                # Log the detected gesture
                print(f"Detected gesture: {detected_gesture}")
                
                # Control Spotify in a separate thread to avoid blocking
                if auth_token:
                    import threading
                    threading.Thread(
                        target=control_spotify,
                        args=(detected_gesture, auth_token)
                    ).start()
                
                return detected_gesture
            
    except Exception as e:
        print(f"Error in gesture detection: {e}")
    
    return current_gesture  # Return current gesture if no new gesture detected

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
    if not token:
        print("Error: No access token provided")
        return False
        
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    base_url = "https://api.spotify.com/v1/me/player"
    
    try:
        # First, check if there's an active device
        try:
            devices_response = requests.get(
                "https://api.spotify.com/v1/me/player/devices",
                headers=headers,
                timeout=3
            )
            
            if devices_response.status_code != 200:
                print(f"Failed to get devices: {devices_response.status_code}")
                if devices_response.status_code == 403:
                    print("Error: Insufficient permissions. Make sure your app has the required scopes.")
                # Continue with basic controls even if device detection fails
                
            devices = devices_response.json().get('devices', [])
            active_devices = [d for d in devices if d.get('is_active', False)]
            
            if not active_devices:
                print("No active Spotify device found. Please make sure Spotify is playing on one of your devices.")
                # Continue with basic controls even if no active device is found
                
        except requests.exceptions.RequestException as e:
            print(f"Device check error: {e}")
            # Continue with basic controls even if device check fails
            
        # Handle volume controls with Spotify Web API
        if action in ["VOLUME_UP", "VOLUME_DOWN"]:
            try:
                # Get current playback state
                response = requests.get(
                    f"{base_url}",
                    headers=headers,
                    timeout=2
                )
                
                if response.status_code != 200:
                    print(f"Failed to get playback state: {response.status_code}")
                    return True
                
                data = response.json()
                
                # Check if device supports volume control
                if not data.get('device', {}).get('supports_volume', True):
                    print("Device doesn't support volume control")
                    return True
                
                # Get current volume or default to 25
                current_volume = data.get('device', {}).get('volume_percent', 25)
                step = 25  # Volume step size
                
                # Calculate new volume (0-100 range)
                if action == "VOLUME_UP":
                    new_volume = min(100, current_volume + step)
                else:  # VOLUME_DOWN
                    new_volume = max(0, current_volume - step)
                
                print(f"Volume: {current_volume}% -> {new_volume}% ({action})")
                
                # Set new volume
                response = requests.put(
                    f"{base_url}/volume",
                    params={"volume_percent": new_volume},
                    headers=headers,
                    timeout=2
                )
                
                if response.status_code not in (200, 202, 204):
                    print(f"Volume control failed: {response.status_code} - {response.text}")
                
                return True
                
            except Exception as e:
                print(f"Volume control error: {str(e)}")
                return True
                
        # Handle other actions
        else:
            try:
                if action in ["NEXT_TRACK", "NEXT"]:
                    response = requests.post(f"{base_url}/next", headers=headers, timeout=5)
                elif action in ["PREVIOUS_TRACK", "PREVIOUS"]:
                    response = requests.post(f"{base_url}/previous", headers=headers, timeout=5)
                elif action in ["PLAY", "PAUSE"]:
                    response = requests.put(f"{base_url}/{action.lower()}", headers=headers, timeout=5)
                elif action == "TOGGLE_PLAY":
                    # First try to play, if that fails, try to pause
                    response = requests.put(f"{base_url}/play", headers=headers, timeout=5)
                    if response.status_code != 204:
                        response = requests.put(f"{base_url}/pause", headers=headers, timeout=5)
                else:
                    print(f"Unknown action: {action}")
                    return False
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed for {action}: {e}")
                return False
        
        # Check if the response was successful
        if response.status_code in (200, 202, 204):
            return True
            
        # Handle specific error cases
        if response.status_code == 401:  # Unauthorized
            print("Error: Invalid or expired token")
        elif response.status_code == 403:  # Forbidden
            print("Error: Insufficient permissions")
        elif response.status_code == 404:  # Not found
            print("Error: No active device found")
        else:
            print(f"Unexpected status code: {response.status_code}")
            
        # Print response details for debugging
        try:
            error_details = response.json()
            print(f"Error details: {error_details}")
        except:
            print(f"Response content: {response.text}")
            
        return False
        
    except Exception as e:
        print(f"Unexpected error in control_spotify: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_frames():
    # Suppress OpenCV warnings
    import os
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'  # Suppress OpenCV warnings
    
    # Try to initialize the camera with retries
    max_retries = 3
    cap = None
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            break
        print(f"Camera initialization attempt {attempt + 1} failed")
        time.sleep(1)
    
    if not cap or not cap.isOpened():
        print("Error: Could not open video device after multiple attempts")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize MediaPipe components with optimized settings
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose, \
        mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2) as hands, \
        mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        
        frame_count = 0
        process_every_n_frame = 2  # Process every 2nd frame for better performance
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while cap.isOpened():
            success = False
            frame = None
            
            # Try reading frame with error handling
            try:
                success, frame = cap.read()
            except Exception as e:
                print(f"Error reading frame: {e}")
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    print("Too many consecutive errors. Exiting video capture.")
                    break
                time.sleep(0.1)
                continue
                
            if not success or frame is None:
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    print("Too many consecutive empty frames. Exiting video capture.")
                    break
                time.sleep(0.1)
                continue
                
            consecutive_errors = 0  # Reset error counter on successful frame read
                
            frame_count += 1
            process_gesture = (frame_count % process_every_n_frame) == 0
            
            # Mirror the frame for more intuitive control
            frame = cv2.flip(frame, 1)
            
            # Process frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            # Process pose, hand, and face landmarks
            pose_results = pose.process(frame_rgb)
            hand_results = hands.process(frame_rgb)
            face_results = face_mesh.process(frame_rgb)
            
            frame_rgb.flags.writeable = True
            
            # Store face landmarks in results for gesture detection
            results = type('', (), {})()
            results.face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
            
            # Update gesture detection in a non-blocking way
            if pose_results.pose_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0] if hand_results.multi_hand_landmarks else None
                # Start a new thread for gesture detection to avoid blocking
                import threading
                threading.Thread(
                    target=detect_gesture,
                    args=(pose_results.pose_landmarks, hand_landmarks, results.face_landmarks)
                ).start()
            
            # Prepare image for display
            frame_rgb.flags.writeable = True
            image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks on every frame for smooth visualization
            # Draw pose landmarks if detected
            if hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    pose_results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
            
            # Draw hand landmarks if detected
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                    )
            
            # Draw face mesh if detected (use a subtle color)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1, circle_radius=1)
                    )
            
            # Draw current gesture status
            with gesture_lock:
                gesture_text = f"Gesture: {current_gesture}" if current_gesture else "No gesture detected"
            
            # Add text to the frame with a background for better visibility
            text_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(image, (5, 5), (15 + text_size[0], 40), (0, 0, 0), -1)  # Background
            cv2.putText(image, gesture_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Encode the frame
            try:
                ret, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error encoding frame: {e}")
                continue
    
    cap.release()

# Routes
@app.route('/')
def index():
    # Check if user is authenticated
    token = request.args.get('access_token') or session.get('spotify_token')
    is_authenticated = False
    
    if token:
        session['spotify_token'] = token
        global auth_token
        auth_token = token
        is_authenticated = True
    
    return render_template('index.html', 
                          auth_url=AUTH_URL, 
                          is_authenticated=is_authenticated,
                          token=token)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/auth')
def auth():
    return jsonify({"auth_url": AUTH_URL})

@app.route('/logout')
def logout():
    # Clear the session data
    session.pop('spotify_token', None)
    session.pop('spotify_refresh', None)
    global auth_token, refresh_token
    auth_token = None
    refresh_token = None
    return redirect(url_for('index'))

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
            session['spotify_token'] = auth_token
            session['spotify_refresh'] = refresh_token
            # Redirect back to the main page with the token in the URL
            return redirect(f'/?access_token={auth_token}')
    
    return "Authentication failed. Please try again."

if __name__ == '__main__':
    # Get initial token if using client credentials flow
    if not auth_token and CLIENT_ID and CLIENT_SECRET:
        auth_token = get_auth_token()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
