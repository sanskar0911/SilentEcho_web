import mediapipe as mp
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
import cv2
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

def load_and_train_model(csv_paths=["gesture_data.csv"]):
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    x = []
    y = []
    
    for path in csv_paths:
        try:
            with open(path, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                
                # Check if first row is a header or data
                first_row = None
                if header:
                    try:
                        # If first element can be float, it's probably data (no header)
                        float(header[1])
                        first_row = header
                    except (ValueError, IndexError):
                        # It's a header, skip it
                        pass
                
                rows_to_process = list(reader)
                if first_row:
                    rows_to_process = [first_row] + rows_to_process

                for row in rows_to_process:
                    if not row: continue
                    label = row[0]
                    values = [float(val) for val in row[1:]]

                    # Normalize landmarks (same as app.py)
                    x_vals = values[:21]
                    y_vals = values[21:]
                    base_x = x_vals[0]
                    base_y = y_vals[0]
                    norm_x = [xv - base_x for xv in x_vals]
                    norm_y = [yv - base_y for yv in y_vals]
                    landmarks = norm_x + norm_y

                    y.append(label)
                    x.append(landmarks)
            print(f"[✅] Loaded training data from: {path}")
        except FileNotFoundError:
            print(f"[⚠️] File '{path}' not found. Skipping.")
        except Exception as e:
            print(f"[❌] Error loading '{path}': {e}")

    if not x:
        print("[⚠️] No training data found. Model not trained.")
        return None, 0.0

    model = RandomForestClassifier(n_estimators=100)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Combined Model Accuracy: ", accuracy)

    return model, accuracy



def predict_gesture(model, landmarks):
    if not model or len(landmarks) != 42:
        return None
    prediction = model.predict([landmarks])[0]
    return prediction


# The following code block is only run when this script is executed directly
if __name__ == "__main__":
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=2)

    cam = cv2.VideoCapture(0)
    sentence = []
    last_prediction_time = 0
    delay_between_signs = 1.5  # seconds

    print("Enter 's' to convert to speech: ")
    print("Enter 'c' to clear sentence: ")

    model, _ = load_and_train_model()  # ⬅️ ignore accuracy here

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
                        break  # Prevent multiple detections in one frame

        # Show last prediction on screen (optional)
        if sentence:
            cv2.putText(frame, f"Last word: {sentence[-1]}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Gesture Recognition", frame)

        if key == 27:  # ESC key
            break
        if key == 115:  # 's' key
            speak = " ".join(sentence)
            print("Speaking:", speak)
        if key == 99:  # 'c' key
            sentence = []
            print("Sentence Cleared.")

    cam.release()
    cv2.destroyAllWindows()
