"""
SilentEcho - Professional Neural Interpreter for Sign Language Recognition

This Flask web application provides real-time sign language recognition using
machine learning models (YOLO and Random Forest classifiers) and computer vision
techniques with MediaPipe for hand tracking.

Features:
- ASL (American Sign Language) recognition
- ISL (Indian Sign Language) recognition for two-handed signs
- Real-time video streaming
- User authentication
- Word and sentence building
- Text-to-speech functionality
- Model training and data collection

Author: SilentEcho Team
"""

from flask import Flask, render_template, Response, jsonify, request, session
from flask_socketio import SocketIO, emit
import cv2
import os
import csv
import subprocess
import time
import sqlite3
import base64
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback-secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Start background loading thread
import threading

@app.route("/health")
def health_check():
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route("/home")
def home():
    from flask import redirect, url_for
    return redirect(url_for("index"))

@app.route('/favicon.ico')
def favicon():
    from flask import send_from_directory
    import os
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.errorhandler(Exception)
def handle_exception(e):
    # Safe fallback for unexpected errors
    return jsonify({"status": "error", "message": "Internal Server Error"}), 500

# =====================
# SQLITE AUTH
# =====================
def init_db():
    """
    Initialize the SQLite database for user authentication.
    Creates a 'users' table with id, email, and password fields.
    """
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()


@app.route("/register", methods=["POST"])
def register():
    """Handle user registration safely."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "email" not in data or "password" not in data:
            return jsonify({"status": "fail", "message": "Invalid payload"})
            
        email = data["email"]
        password = generate_password_hash(data["password"])

        conn = sqlite3.connect("users.db")
        c = conn.cursor()

        try:
            c.execute("INSERT INTO users(email,password) VALUES (?,?)", (email, password))
            conn.commit()
        except sqlite3.IntegrityError:
            return jsonify({"status": "exists"})
        finally:
            conn.close()

        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"Registration Error: {e}")
        return jsonify({"status": "fail"})


@app.route("/login", methods=["POST"])
def login():
    """Handle user login safely."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "email" not in data or "password" not in data:
            return jsonify({"status": "fail", "message": "Invalid payload"})

        conn = sqlite3.connect("users.db")
        c = conn.cursor()

        c.execute("SELECT * FROM users WHERE email=?", (data["email"],))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], data["password"]):
            session["user"] = data["email"]
            return jsonify({"status": "ok"})

        return jsonify({"status": "fail"})
    except Exception as e:
        print(f"Login Error: {e}")
        return jsonify({"status": "fail"})


# =====================
# LOAD MODELS
# =====================
asl_model = None
current_accuracy = 0
isl_model = None
yolo_model = None

mp_hands = None
hands = None
mp_draw = None

models_loaded = False

def load_models_if_needed():
    global asl_model, current_accuracy, isl_model, yolo_model, hands, models_loaded, mp_hands, mp_draw
    if models_loaded:
        return
        
    print("Loading ML models...")
    
    # Deferred heavy imports
    import mediapipe as mp
    import torch
    import gc
    from ultralytics import YOLO
    from gesture_recognition import load_and_train_model
    
    # Aggressive Memory Optimization for Render Free Tier (512MB RAM Limit)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    
    try:
        asl_model, current_accuracy = load_and_train_model()
    except Exception as e:
        print("Failed to load ASL model:", e)
        
    try:
        isl_model, _ = load_and_train_model("isl_gesture_data.csv")
    except Exception as e:
        print("Failed to load ISL model:", e)
        
    try:
        yolo_model = YOLO("runs/classify/train3/weights/best.pt")
    except:
        try:
            yolo_model = YOLO("best.pt")
        except:
            print("WARNING: YOLO model not found")
            yolo_model = None
            
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    models_loaded = True
    
    # Force garbage collection to free any temporary memory spikes during loading
    gc.collect()
    
    print("Models loaded successfully.")

# Spawn background loading thread
threading.Thread(target=load_models_if_needed, daemon=True).start()


# =====================
# GLOBAL STATE
# =====================
camera=None  # Video capture object
latest_prediction=""  # Current recognized gesture
latest_confidence=0   # Confidence score of prediction
last_prediction=None  # Previous prediction for debouncing
last_spoken=0  # Timestamp of last spoken prediction
speak_enabled=True  # Toggle for text-to-speech
current_inference_mode="auto"  # "yolo", "mediapipe", or "auto"

# Word mode variables
word_mode=False
word_buffer=[]  # Letters collected for word building
last_added_letter=None

# Sentence building features
sentence_buffer=[]  # Words collected for sentence
conversation_history=[]  # History of conversations
language="en"  # Language for TTS


# =====================
# VIDEO STREAM
# =====================
@socketio.on('image')
def process_image(data):
    """
    WebSocket event listener for video frames.
    Receives base64 image from client, performs gesture recognition,
    and emits the annotated frame and prediction back.
    """
    global latest_prediction, last_prediction, last_spoken, latest_confidence
    global word_mode, word_buffer, last_added_letter, current_inference_mode, yolo_model
    global asl_model, isl_model, hands, mp_hands, mp_draw
    
    # If background thread hasn't finished loading ML, return the original frame immediately
    if not models_loaded:
        emit('processed_image', {'image': data, 'prediction': 'INITIALIZING ML...', 'confidence': 0})
        return
        
    # Import locally to avoid global blocking
    from gesture_recognition import predict_gesture

    try:
        # Decode base64 image
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        prediction = ""
        confidence = 0
        yolo_pred = None
        yolo_conf = 0
        use_mediapipe = False
        
        if current_inference_mode in ["yolo", "auto"] and yolo_model:
            results = yolo_model.predict(frame, imgsz=160, verbose=False)
            if len(results) > 0:
                probs = results[0].probs
                if probs is not None:
                    for idx in probs.top5:
                        class_name = results[0].names[int(idx)]
                        if len(class_name) == 1 and class_name.isalpha():
                            yolo_pred = class_name
                            yolo_conf = float(probs.data[int(idx)]) * 100.0
                            break
            
            if yolo_pred and (current_inference_mode == "yolo" or (current_inference_mode == "auto" and yolo_conf >= 60.0)):
                prediction = yolo_pred
                confidence = round(yolo_conf, 2)
            else:
                use_mediapipe = True
        else:
            use_mediapipe = True

        if use_mediapipe:
            result = hands.process(rgb)
            asl_pred = None
            isl_pred = None

            if result.multi_hand_landmarks:
                num_hands = len(result.multi_hand_landmarks)

                cv2.putText(frame, f"Hands: {num_hands}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                # ASL
                h = result.multi_hand_landmarks[0]
                x = [lm.x for lm in h.landmark]
                y = [lm.y for lm in h.landmark]
                bx = x[0]; by = y[0]
                land = [(v - bx) for v in x] + [(v - by) for v in y]

                if asl_model:
                    probs = asl_model.predict_proba([land])[0]
                    max_index = probs.argmax()
                    asl_pred = asl_model.classes_[max_index]
                    confidence = round(probs[max_index] * 100, 2)

                # ISL
                if num_hands == 2:
                    def norm(h):
                        x = [lm.x for lm in h.landmark]
                        y = [lm.y for lm in h.landmark]
                        bx = x[0]; by = y[0]
                        return [(v - bx) for v in x] + [(v - by) for v in y]

                    land2 = norm(result.multi_hand_landmarks[0]) + \
                            norm(result.multi_hand_landmarks[1])

                    if isl_model:
                        probs = isl_model.predict_proba([land2])[0]
                        max_index = probs.argmax()
                        isl_pred = isl_model.classes_[max_index]
                        confidence = round(probs[max_index] * 100, 2)

                if isl_pred:
                    prediction = isl_pred
                elif asl_pred:
                    prediction = asl_pred

        if prediction:
            if prediction == last_prediction and time.time() - last_spoken > 1.2:
                latest_prediction = prediction
                latest_confidence = confidence

                if word_mode:
                    if prediction != last_added_letter:
                        word_buffer.append(prediction)
                        last_added_letter = prediction

                last_spoken = time.time()

            else:
                last_prediction = prediction

        cv2.putText(frame, f"Mode: {current_inference_mode.upper()}", (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"{latest_prediction} ({latest_confidence}%)", (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        emit('processed_image', {
            'image': f'data:image/jpeg;base64,{encoded_img}',
            'prediction': latest_prediction,
            'confidence': latest_confidence
        })

    except Exception as e:
        print(f"Error processing image: {e}")

# =====================
# ROUTES
# =====================
@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/get_prediction")
def get_prediction():
    """Return current prediction and confidence as JSON."""
    return jsonify({
        "prediction":latest_prediction,
        "confidence":latest_confidence
    })

@app.route("/set_mode", methods=["POST"])
def set_mode():
    """Set the inference mode (yolo, mediapipe, auto)."""
    global current_inference_mode
    mode = request.get_json().get("mode", "auto")
    if mode in ["yolo", "mediapipe", "auto"]:
        current_inference_mode = mode
    return jsonify({"mode": current_inference_mode})

@app.route("/toggle_speak", methods=["POST"])
def toggle_speak():
    """Toggle text-to-speech functionality."""
    global speak_enabled
    speak_enabled=request.get_json().get("enabled",True)
    return jsonify({"speak_enabled":speak_enabled})


# =====================
# WORD MODE
# =====================
@app.route("/start_word_mode",methods=["POST"])
def start_word_mode():
    """Start word building mode."""
    global word_mode,word_buffer,last_added_letter
    word_mode=True
    word_buffer=[]
    last_added_letter=None
    return jsonify({"msg":"started"})


@app.route("/finish_word",methods=["POST"])
def finish_word():
    """Finish word building and optionally speak it."""
    global word_mode,last_added_letter
    word_mode=False
    final="".join(word_buffer)
    last_added_letter=None

    audio_url = None
    if speak_enabled and final:
        try:
            from gtts import gTTS
            tts = gTTS(text=final, lang='en')
            os.makedirs("static", exist_ok=True)
            tts.save("static/speech.mp3")
            audio_url = "/static/speech.mp3"
        except Exception as e:
            print("TTS Error:", e)

    return jsonify({"word": final, "audio_url": audio_url})


@app.route("/delete_letter",methods=["POST"])
def delete_letter():
    """Delete the last letter from word buffer."""
    global word_buffer
    if word_buffer:
        word_buffer.pop()
    return jsonify({"word":"".join(word_buffer)})


# =====================
# NEW FEATURES
# =====================

@app.route("/add_word", methods=["POST"])
def add_word():
    """Add current word buffer to sentence buffer."""
    global sentence_buffer, word_buffer

    word="".join(word_buffer)

    if word:
        sentence_buffer.append(word)

    word_buffer=[]

    return jsonify({"sentence":" ".join(sentence_buffer)})


@app.route("/speak_sentence", methods=["POST"])
def speak_sentence():
    """Speak the current sentence buffer."""
    global sentence_buffer

    sentence=" ".join(sentence_buffer)

    audio_url = None
    if sentence:
        try:
            from gtts import gTTS
            tts = gTTS(text=sentence, lang='en')
            os.makedirs("static", exist_ok=True)
            tts.save("static/speech.mp3")
            audio_url = "/static/speech.mp3"
        except Exception as e:
            print("TTS Error:", e)

    return jsonify({"sentence": sentence, "audio_url": audio_url})


@app.route("/get_history")
def get_history():
    """Return conversation history."""
    return jsonify({"history":conversation_history})


@app.route("/get_accuracy")
def get_accuracy():
    """Return current model accuracy safely."""
    try:
        acc = current_accuracy if current_accuracy else 0
        return jsonify({"accuracy": round(acc * 100, 2)})
    except Exception:
        return jsonify({"accuracy": 0})


@app.route("/set_language", methods=["POST"])
def set_language():
    """Set language for text-to-speech."""
    global language
    language=request.json.get("lang","en")
    return jsonify({"lang":language})


# =====================
# TRAIN MODEL
# =====================
@app.route("/train_model", methods=["POST"])
def train_model():
    """Retrain the ASL model with current data."""
    global asl_model,current_accuracy
    asl_model,current_accuracy=load_and_train_model()
    return jsonify({"accuracy":round(current_accuracy*100,2)})


# =====================
# START COLLECTION
# =====================
@app.route("/start_collection", methods=["POST"])
def start_collection():
    """
    Start collecting gesture data for a new label.
    Collects 100 samples of hand landmarks.
    """

    global camera
    label=request.json.get("label","").strip()
    if not label:
        return jsonify({"error":"Label required"})

    filename="gesture_data.csv"
    file_exists=os.path.isfile(filename)

    samples=0

    while samples<100:

        if camera is None or not camera.isOpened():
            camera=cv2.VideoCapture(0)

        ret,frame=camera.read()
        if not ret:
            continue

        frame=cv2.flip(frame,1)

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=hands.process(rgb)

        if result.multi_hand_landmarks:

            h=result.multi_hand_landmarks[0]

            x=[lm.x for lm in h.landmark]
            y=[lm.y for lm in h.landmark]

            bx=x[0];by=y[0]

            land=[(v-bx) for v in x]+[(v-by) for v in y]

            if len(land)==42:

                with open(filename,"a",newline="") as f:

                    w=csv.writer(f)

                    if not file_exists:
                        w.writerow(["label"]+[f"f{i}" for i in range(42)])
                        file_exists=True

                    w.writerow([label]+land)

                    samples+=1

        cv2.waitKey(10)

    return jsonify({"samples":samples})


# =====================
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    # allow_unsafe_werkzeug required when running socketio without async framework
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)