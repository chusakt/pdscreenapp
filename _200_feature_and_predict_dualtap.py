import json
import math
import numpy as np
from statistics import mean
import os
import pickle

from config import base_path
modeltouse = "trained_model_dualtap__SMOTE.pkl"
new_file_path_modeltouse = os.path.join(base_path, modeltouse)

# Load the trained model from the pickle file
with open(new_file_path_modeltouse, 'rb') as f:
    model = pickle.load(f)


# Function to extract TapCount and additional statistics from the JSON file
def extract_tapcount_statistics(json_file):
    print(f"Processing file: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    score = data['score']

    # -------------- about circle ---------------
    jL = data['recording']['circleL']
    jR = data['recording']['circleR']
    cLx, cLy = jL['x'], jL['y']
    cRx, cRy = jR['x'], jR['y']
    cRadius = jL['r']

    distancEachPoint = []
    inLeft = 0
    inRight = 0
    jf = data['recording']['taps']
    ts_instroke_mean = []

    for jstroke in jf:
        jjj = jstroke['data']
        ts_instroke = []
        for jjjj in jjj:
            xpo, ypo, ts_ = jjjj['x'], jjjj['y'], jjjj['ts']
            ts_instroke.append(ts_)

            # Compare to L and R, then decide which side
            if (xpo - cLx) >= (cRx - xpo):
                inRight += 1
                dispo = math.sqrt((xpo - cRx) ** 2 + (ypo - cRy) ** 2)
                distancEachPoint.append(dispo)
            else:
                inLeft += 1
                dispo = math.sqrt((xpo - cLx) ** 2 + (ypo - cLy) ** 2)
                distancEachPoint.append(dispo)

        ts_instroke_mean.append(mean(ts_instroke))

    countinside = sum(map(lambda x: x < cRadius, distancEachPoint))
    countall = len(distancEachPoint)
    ppInsideToAll = countinside / countall if countall > 0 else 0

    try:
        if inLeft / inRight > 1:
            ppLeftToRight = inRight / inLeft
        else:
            ppLeftToRight = inLeft / inRight
    except ZeroDivisionError:
        ppLeftToRight = 0

    tDiff = [item1 - item2 for item1, item2 in zip(ts_instroke_mean[1:], ts_instroke_mean[:-1])]
    tDiff_mean = mean(tDiff) if tDiff else 0
    tDiff_max = max(tDiff) if tDiff else 0
    tDiff_min = min(tDiff) if tDiff else 0

    tap_points_list = [len(tap['data']) for tap in jf]
    maxTapPoints = np.max(tap_points_list) if tap_points_list else 0
    stdTapPoints = np.std(tap_points_list) if tap_points_list else 0
    meanTapPoints = np.mean(tap_points_list) if tap_points_list else 0
    tapCount = len(tap_points_list)

    features = {
        "Score": score,
        "Tap Count": tapCount,
        "Max Tap Points": maxTapPoints,
        "Std Tap Points": stdTapPoints,
        "Mean Tap Points": meanTapPoints,
        "Total Points": countall,
        "ppInsideToAll": ppInsideToAll,
        "ppLeftToRight": ppLeftToRight,
        "tDiff_mean": tDiff_mean,
        "tDiff_max": tDiff_max,
        "tDiff_min": tDiff_min
    }

    return features


#------------------------------------------------------------

# Example usage with a JSON file
json_file_path = "./datajson/control/dualTap_1721110619936.json"  # Update with the actual JSON file path

# Extract features from the JSON file
features = extract_tapcount_statistics(json_file_path)

# Print the extracted features
print("Extracted Features:")
for feature, value in features.items():
    print(f"{feature}: {value}")

# Convert dictionary values to a flat list
features_ = list(features.values())

# Convert to a 2D NumPy array
X_test = np.array(features_).reshape(1, -1)

# Predict on the test set
y_pred = model.predict(X_test)

# print output compare to target

print('predicted:',y_pred)