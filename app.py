from flask import Flask, render_template, jsonify, request
import csv
import os
from gesture_recognition import load_and_train_model, predict_gesture

app = Flask(__name__)

# Load model at startup
model, current_accuracy = load_and_train_model()

latest_prediction = "No gesture detected"
speak_enabled = True


@app.route('/')
def index():
    return render_template('index.html')


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
        filename = "gesture_data.csv"
        file_exists = os.path.isfile(filename)

        with open(filename, "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["label"] + [f"f{i}" for i in range(42)])
            print(f"[ℹ️] Ready to collect data for '{label}' (client should handle actual capture).")

        return jsonify({"status": "success", "message": f"Ready to collect data for '{label}'."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed: {str(e)}"})


# ---------- Aliases so frontend matches ----------
@app.route('/train_model', methods=['POST'])
def train_model():
    return train_model_csv()

@app.route('/model_accuracy')
def model_accuracy():
    return get_accuracy()
# -------------------------------------------------


if __name__ == "__main__":
    # Hugging Face requires app to run on host 0.0.0.0 and port 7860
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Flask server on http://0.0.0.0:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False)
