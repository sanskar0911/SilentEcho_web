from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import os
import csv
import subprocess
import time
from gesture_recognition import load_and_train_model, predict_gesture

app = Flask(__name__)

# Load model and accuracy
model, current_accuracy = load_and_train_model()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Globals
latest_prediction = "No gesture detected"
latest_landmarks = []
last_prediction = None
speak_enabled = True
last_spoken = 0

# === Word Mode Feature ===
word_mode = False
word_buffer = []


def gen_frames():
    global latest_prediction, latest_landmarks, last_prediction, last_spoken, word_mode, word_buffer
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                x_vals = [lm.x - base_x for lm in hand_landmarks.landmark]
                y_vals = [lm.y - base_y for lm in hand_landmarks.landmark]
                landmarks = x_vals + y_vals
                latest_landmarks = landmarks
                prediction = predict_gesture(model, landmarks)

                if prediction:
                    if prediction == last_prediction:
                        if time.time() - last_spoken > 1.5:
                            latest_prediction = prediction
                            if word_mode:
                                word_buffer.append(prediction)
                            if speak_enabled:
                                subprocess.Popen([
                                    "python",
                                    "-c",
                                    f"import pyttsx3; e=pyttsx3.init(); e.say('{prediction}'); e.runAndWait()"
                                ])
                            last_spoken = time.time()
                    else:
                        last_prediction = prediction
        else:
            latest_prediction = "No gesture detected"
            latest_landmarks = []
            last_prediction = None

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# === ROUTES ===
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_prediction')
def get_prediction():
    return jsonify({"prediction": latest_prediction})


@app.route('/get_accuracy')
def get_accuracy():
    return jsonify({"accuracy": round(current_accuracy * 100, 2)})


@app.route('/train_model_csv', methods=['POST'])
def train_model_csv():
    global model, current_accuracy
    try:
        print("[🧠] Training model from CSV...")
        model, current_accuracy = load_and_train_model()
        print("Accuracy:", current_accuracy)
        return jsonify({"message": f"Training complete. Accuracy: {round(current_accuracy * 100, 2)}%"})
    except Exception as e:
        print("[❌] Training failed:", e)
        return jsonify({"message": "Training failed", "error": str(e)})


@app.route('/train_model', methods=['POST'])
def train_model():
    return train_model_csv()


@app.route('/toggle_speak', methods=['POST'])
def toggle_speak():
    global speak_enabled
    data = request.get_json()
    speak_enabled = data.get('enabled', True)
    return jsonify({"speak_enabled": speak_enabled})


@app.route('/start_collection', methods=['POST'])
def start_collection():
    label = request.json.get("label", "").strip()
    if not label:
        return jsonify({"status": "error", "message": "Label is required."})

    try:
        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                               min_detection_confidence=0.7, min_tracking_confidence=0.7)
        mp_draw = mp.solutions.drawing_utils

        filename = "gesture_data.csv"
        file_exists = os.path.isfile(filename)
        samples = 0
        max_samples = 100

        while samples < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    x_vals = [lm.x for lm in hand_landmarks.landmark]
                    y_vals = [lm.y for lm in hand_landmarks.landmark]
                    base_x = x_vals[0]
                    base_y = y_vals[0]
                    norm_x = [x - base_x for x in x_vals]
                    norm_y = [y - base_y for y in y_vals]
                    landmarks = norm_x + norm_y
                    if len(landmarks) == 42:
                        with open(filename, "a", newline='') as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(["label"] + [f"f{i}" for i in range(42)])
                                file_exists = True
                            writer.writerow([label] + landmarks)
                        samples += 1
                        print(f"[+] Collected sample {samples} for '{label}'.")
            cv2.waitKey(10)

        cap.release()
        return jsonify({"status": "success", "message": f"Collected {samples} samples for '{label}'."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to collect data: {str(e)}"})


@app.route('/record_gesture', methods=['POST'])
def record_gesture():
    return start_collection()


@app.route('/model_accuracy')
def model_accuracy():
    return get_accuracy()


# === WORD MODE ROUTES ===
@app.route('/start_word_mode', methods=['POST'])
def start_word_mode():
    global word_mode, word_buffer
    word_mode = True
    word_buffer = []
    return jsonify({"message": "Word mode started"})


@app.route('/finish_word', methods=['POST'])
def finish_word():
    global word_mode, word_buffer
    word_mode = False
    final_word = "".join(word_buffer)
    return jsonify({"word": final_word})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
