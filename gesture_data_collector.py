import mediapipe as mp
import cv2
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pyttsx3
import time

engine = pyttsx3.init()

def load_and_train_model(csv_path="gesture_data.csv"):
    x = []
    y = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) != 43 or any(val.strip() == '' for val in row[1:]):
                    continue  # Skip rows with missing/invalid data

                try:
                    values = [float(val) for val in row[1:]]
                except ValueError:
                    continue  # Skip rows that can't convert to float

                # Normalize landmarks (same as app.py)
                x_vals = values[:21]
                y_vals = values[21:]
                base_x = x_vals[0]
                base_y = y_vals[0]
                norm_x = [x - base_x for x in x_vals]
                norm_y = [y - base_y for y in y_vals]
                landmarks = norm_x + norm_y

                y.append(row[0])
                x.append(landmarks)
    except FileNotFoundError:
        print(f"[⚠️] File '{csv_path}' not found. Model not trained.")
        return None, 0.0

    if not x:
        print("[❌] No valid training data found.")
        return None, 0.0

    model = RandomForestClassifier(n_estimators=100)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Accuracy: ", accuracy)

    return model, accuracy


def predict_gesture(model, landmarks):
    if not model or len(landmarks) != 42:
        return None
    prediction = model.predict([landmarks])[0]
    return prediction


if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=2)
    mp_draw = mp.solutions.drawing_utils

    cam = cv2.VideoCapture(0)
    sentence = []
    last_prediction_time = 0
    delay_between_signs = 1.5  # seconds

    print("Enter 's' to convert to speech: ")
    print("Enter 'c' to clear sentence: ")

    model, _ = load_and_train_model()

    while cam.isOpened():
        key = cv2.waitKey(1)
        success, frame = cam.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img)

        current_time = time.time()
        if result.multi_hand_landmarks and (current_time - last_prediction_time) > delay_between_signs:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append(lm.x)
                for lm in hand_landmarks.landmark:
                    landmark_list.append(lm.y)

                if len(landmark_list) == 42:
                    prediction = predict_gesture(model, landmark_list)
                    if prediction:
                        sentence.append(prediction)
                        print("Captured word:", prediction)
                        last_prediction_time = current_time
                        break

        if sentence:
            cv2.putText(frame, f"Last word: {sentence[-1]}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Gesture Recognition", frame)

        if key == 27:
            break
        if key == 115:
            speak = " ".join(sentence)
            print("Speaking:", speak)
            engine.say(speak)
            engine.runAndWait()
        if key == 99:
            sentence = []
            print("Sentence Cleared.")

    cam.release()
    cv2.destroyAllWindows()
