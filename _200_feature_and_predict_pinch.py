import os
import json
import numpy as np
import os
import pickle

from config import base_path
modeltouse = "trained_model_pinch_fe_SMOTE.pkl"
new_file_path_modeltouse = os.path.join(base_path, modeltouse)

# Load the trained model from the pickle file
with open(new_file_path_modeltouse, 'rb') as f:
    model = pickle.load(f)

# # Define base path and output file
# from config import base_path

# Function to calculate moving average
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Function to process a single pinch-like JSON file and compute the required metrics
def process_pinch_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize variables and lists for calculations
    extracted_values = []
    CountAllHandOff = 0
    allSignSwitchCount = 0
    x_mts_start_stack = []
    x_mts_end_stack = []
    x_norm_std_stack = []
    x_norm_mean_stack = []
    x_mx1_std_stack = []
    x_mx1_mean_stack = []
    x_mx2_std_stack = []
    x_mx2_mean_stack = []
    x_my1_std_stack = []
    x_my1_mean_stack = []
    x_my2_std_stack = []
    x_my2_mean_stack = []

    # Check if 'data' exists in the JSON structure
    if 'data' in data:
        for indata in data['data']:
            if 'meta' in indata:  # If 'meta' exists, extract "target" and "absoluteTime"
                j_meta = indata['meta']

                # Initialize variables to store calculations
                mts, mx1, mx2, my1, my2, mdia = [], [], [], [], [], []

                # Check if 'data' key exists next to 'meta'
                if 'data' in indata:
                    j_indata = indata['data']

                    # Perform calculations based on the data
                    for k_injdata in j_indata:
                        CountAllHandOff += 1
                        for m_inkdata in k_injdata:
                            mts.append(m_inkdata['timestamp'])
                            mx1.append(m_inkdata['x1'])
                            mx2.append(m_inkdata['x2'])
                            my1.append(m_inkdata['y1'])
                            my2.append(m_inkdata['y2'])
                            mdia.append(m_inkdata['diameter'])

                    # Smoothing and sign-switch counting for mdia
                    mvWindow = 25
                    if len(mdia) > mvWindow + 1:
                        mav = moving_average(mdia, mvWindow)
                        signSwitchCount = 0
                        for idx, dat_ in enumerate(mav):
                            if idx == 0:
                                mvToRaw_a = (dat_ > mdia[idx])
                            else:
                                mvToRaw_b = (dat_ > mdia[idx])
                                if mvToRaw_b != mvToRaw_a:
                                    signSwitchCount += 1
                                    mvToRaw_a = mvToRaw_b
                        allSignSwitchCount += signSwitchCount

                    # Perform normalization and additional calculations
                    mts_start = mts[0]
                    mts_end = mts[-1]

                    mdia_ar = np.array(mdia)
                    if np.max(mdia_ar) != np.min(mdia_ar):
                        x_norm_ar = (mdia_ar - np.min(mdia_ar)) / (np.max(mdia_ar) - np.min(mdia_ar))
                    else:
                        x_norm_ar = np.zeros_like(mdia_ar)
                    x_norm_std = np.std(x_norm_ar)
                    x_norm_mean = np.mean(x_norm_ar)

                    x_mx1_std = np.std(mx1)
                    x_mx1_mean = np.mean(mx1)

                    x_mx2_std = np.std(mx2)
                    x_mx2_mean = np.mean(mx2)

                    x_my1_std = np.std(my1)
                    x_my1_mean = np.mean(my1)

                    x_my2_std = np.std(my2)
                    x_my2_mean = np.mean(my2)

                    # Append values to overall stacks
                    x_mts_start_stack.append(mts_start)
                    x_mts_end_stack.append(mts_end)

                    x_norm_std_stack.append(x_norm_std)
                    x_norm_mean_stack.append(x_norm_mean)

                    x_mx1_std_stack.append(x_mx1_std)
                    x_mx1_mean_stack.append(x_mx1_mean)

                    x_mx2_std_stack.append(x_mx2_std)
                    x_mx2_mean_stack.append(x_mx2_mean)

                    x_my1_std_stack.append(x_my1_std)
                    x_my1_mean_stack.append(x_my1_mean)

                    x_my2_std_stack.append(x_my2_std)
                    x_my2_mean_stack.append(x_my2_mean)

    # Final calculations
    mts_start_mean = np.mean(x_mts_start_stack)
    mts_start_max = np.max(x_mts_start_stack)
    mts_range_max = np.max(x_mts_end_stack)

    mdia_std = np.mean(x_norm_std_stack)
    mdia_mean = np.mean(x_norm_mean_stack)

    mx1_std = np.mean(x_mx1_std_stack)
    mx1_mean = np.mean(x_mx1_mean_stack)

    mx2_std = np.mean(x_mx2_std_stack)
    mx2_mean = np.mean(x_mx2_mean_stack)

    my1_std = np.mean(x_my1_std_stack)
    my1_mean = np.mean(x_my1_mean_stack)

    my2_std = np.mean(x_my2_std_stack)
    my2_mean = np.mean(x_my2_mean_stack)

    # Collect the final metrics and return them as a dictionary
    features = {
        "mdia_std": mdia_std,
        "mdia_mean": mdia_mean,
        "mx1_std": mx1_std,
        "mx1_mean": mx1_mean,
        "mx2_std": mx2_std,
        "mx2_mean": mx2_mean,
        "my1_std": my1_std,
        "my1_mean": my1_mean,
        "my2_std": my2_std,
        "my2_mean": my2_mean,
        "CountAllHandOff": CountAllHandOff,
        "allSignSwitchCount": allSignSwitchCount,
        "mts_start_mean": mts_start_mean,
        "mts_range_max": mts_range_max,
        "mts_start_max": mts_start_max
    }
                
    return features


#------------------------------------------------------------

# Example usage with a single JSON file
json_file_path = "./datajson/control/pinchToSize_1721110619936.json"  # Update this path with the actual JSON file location

# Extract features from the JSON file
extracted_features = process_pinch_file(json_file_path)

# Print the extracted features
print("\nExtracted Features:")
for feature, value in extracted_features.items():
    print(f"{feature}: {value}")


# Convert dictionary values to a flat list
features_ = list(extracted_features.values())

# Convert to a 2D NumPy array
X_test = np.array(features_).reshape(1, -1)

# Predict on the test set
y_pred = model.predict(X_test)

# print output compare to target

print('predicted:',y_pred)