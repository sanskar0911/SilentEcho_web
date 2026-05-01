import csv
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_acc(csv_path):
    x = []
    y = []
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 43: continue
            label = row[0]
            values = [float(val) for val in row[1:]]
            x_vals = values[:21]
            y_vals = values[21:]
            base_x, base_y = x_vals[0], y_vals[0]
            landmarks = [x - base_x for x in x_vals] + [y - base_y for y in y_vals]
            y.append(label)
            x.append(landmarks)
    
    if not x: return 0.0
    model = RandomForestClassifier(n_estimators=100)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

print(f"ASL Accuracy: {get_acc('gesture_data.csv')*100:.2f}%")
print(f"ISL Accuracy: {get_acc('isl_gesture_data.csv')*100:.2f}%")
print(f"Pretrained_word: {get_acc('pretrained_word.csv')*100:.2f}%")
