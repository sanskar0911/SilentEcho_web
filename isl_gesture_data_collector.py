import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mediapipe as mp
import cv2
import csv
import time
import pyttsx3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

engine = pyttsx3.init()

CSV_PATH = "isl_gesture_data.csv"
SAMPLES_PER_LABEL = 500


# =========================
# Create CSV Header
# =========================
def create_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"]

            for i in range(21):
                header.append(f"Lx{i}")
            for i in range(21):
                header.append(f"Ly{i}")

            for i in range(21):
                header.append(f"Rx{i}")
            for i in range(21):
                header.append(f"Ry{i}")

            writer.writerow(header)


# =========================
# Normalize Two Hands
# =========================
def normalize_two_hands(hand1, hand2):

    def normalize(hand):
        x_vals = hand[:21]
        y_vals = hand[21:]
        base_x = x_vals[0]
        base_y = y_vals[0]
        norm_x = [x - base_x for x in x_vals]
        norm_y = [y - base_y for y in y_vals]
        return norm_x + norm_y

    return normalize(hand1) + normalize(hand2)


# =========================
# Collect Data
# =========================
def collect_data(cam, hands):
    label = input("\nEnter ISL label to train (Two Hands): ").strip()
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

            if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:

                hand1 = []
                hand2 = []

                for lm in result.multi_hand_landmarks[0].landmark:
                    hand1.append(lm.x)
                for lm in result.multi_hand_landmarks[0].landmark:
                    hand1.append(lm.y)

                for lm in result.multi_hand_landmarks[1].landmark:
                    hand2.append(lm.x)
                for lm in result.multi_hand_landmarks[1].landmark:
                    hand2.append(lm.y)

                if len(hand1) == 42 and len(hand2) == 42:
                    normalized = normalize_two_hands(hand1, hand2)
                    writer.writerow([label] + normalized)
                    count += 1
                    print(f"Collected: {count}/{SAMPLES_PER_LABEL}", end="\r")

            cv2.putText(frame, f"Show BOTH hands: {count}/{SAMPLES_PER_LABEL}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

            cv2.imshow("ISL Two Hand Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    print("\n[✅] Two-Hand Data Collection Completed!")


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
            if len(row) != 85:
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

    model = RandomForestClassifier(n_estimators=200)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"\n🎯 ISL Model Accuracy: {round(acc*100,2)}%")

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
        max_num_hands=2
    )

    cam = cv2.VideoCapture(0)

    print("\nClick camera window then press:")
    print("T → Train new ISL label")
    print("ESC → Exit\n")

    model = None

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("ISL Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('t'):
            collect_data(cam, hands)
            model = load_and_train_model()

        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
