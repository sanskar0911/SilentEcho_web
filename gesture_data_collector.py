import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide TensorFlow warnings

import mediapipe as mp
import cv2
import csv
import time
import pyttsx3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

engine = pyttsx3.init()

CSV_PATH = "gesture_data.csv"
SAMPLES_PER_LABEL = 5000


# =========================
# Create CSV Header
# =========================
def create_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + \
                     [f"x{i}" for i in range(21)] + \
                     [f"y{i}" for i in range(21)]
            writer.writerow(header)


# =========================
# Normalize landmarks
# =========================
def normalize_landmarks(landmarks):
    x_vals = landmarks[:21]
    y_vals = landmarks[21:]
    base_x = x_vals[0]
    base_y = y_vals[0]
    norm_x = [x - base_x for x in x_vals]
    norm_y = [y - base_y for y in y_vals]
    return norm_x + norm_y


# =========================
# Collect Data
# =========================
def collect_data(cam, hands):
    label = input("\nEnter label name to train: ").strip()
    print(f"[INFO] Collecting {SAMPLES_PER_LABEL} samples for '{label}'")

    count = 0
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        while count < SAMPLES_PER_LABEL:
            success, frame = cam.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(img)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmark_list = []
                    for lm in hand_landmarks.landmark:
                        landmark_list.append(lm.x)
                    for lm in hand_landmarks.landmark:
                        landmark_list.append(lm.y)

                    if len(landmark_list) == 42:
                        normalized = normalize_landmarks(landmark_list)
                        writer.writerow([label] + normalized)
                        count += 1
                        print(f"Collected: {count}/{SAMPLES_PER_LABEL}", end="\r")

            cv2.putText(frame, f"Collecting: {count}/{SAMPLES_PER_LABEL}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

            cv2.imshow("Collecting Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    print("\n[✅] Data Collection Completed!")


# =========================
# Train Model
# =========================
def load_and_train_model():
    x = []
    y = []

    if not os.path.exists(CSV_PATH):
        print("[❌] Dataset not found.")
        return None

    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if len(row) != 43:
                continue
            try:
                values = [float(val) for val in row[1:]]
            except:
                continue

            x.append(values)
            y.append(row[0])

    if not x:
        print("[❌] Dataset empty.")
        return None

    model = RandomForestClassifier(n_estimators=150)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"[🎯] Model Accuracy: {round(acc*100,2)}%")

    return model


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    create_csv()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        max_num_hands=1
    )

    mp_draw = mp.solutions.drawing_utils
    cam = cv2.VideoCapture(0)

    print("\nClick on camera window then press:")
    print("T → Train new label")
    print("S → Speak sentence")
    print("C → Clear sentence")
    print("ESC → Exit\n")

    model = None
    sentence = []
    last_prediction_time = 0
    delay = 1.5

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img)

        cv2.imshow("Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF   # FIXED KEY HANDLING

        # TRAIN
        if key == ord('t'):
            collect_data(cam, hands)
            model = load_and_train_model()

        # PREDICT
        current_time = time.time()
        if model and result.multi_hand_landmarks and \
                (current_time - last_prediction_time) > delay:

            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append(lm.x)
                for lm in hand_landmarks.landmark:
                    landmark_list.append(lm.y)

                if len(landmark_list) == 42:
                    normalized = normalize_landmarks(landmark_list)
                    prediction = model.predict([normalized])[0]
                    sentence.append(prediction)
                    print("Captured:", prediction)
                    last_prediction_time = current_time
                    break

        # DISPLAY TEXT
        if sentence:
            cv2.putText(frame, f"Sentence: {' '.join(sentence)}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        # SPEAK
        if key == ord('s') and sentence:
            speak = " ".join(sentence)
            print("Speaking:", speak)
            engine.say(speak)
            engine.runAndWait()

        # CLEAR
        if key == ord('c'):
            sentence = []
            print("Sentence Cleared.")

        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
