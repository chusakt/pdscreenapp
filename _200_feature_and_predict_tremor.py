import os
import json
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
import EntropyHub as EH
import os
import json
import numpy as np
import os
import pickle

from config import base_path
modeltouse = "trained_model_RestHand_featu_SMOTE.pkl"
new_file_path_modeltouse = os.path.join(base_path, modeltouse)

# Load the trained model from the pickle file
with open(new_file_path_modeltouse, 'rb') as f:
    model = pickle.load(f)

# Sampling rate (10 Hz)
sampling_rate = 10.0
nyquist_freq = sampling_rate / 2.0  # Nyquist frequency is half of the sampling rate
cutoff_freq = 0.12  # Desired cutoff frequency (0.12 Hz)
normalized_cutoff = cutoff_freq / nyquist_freq  # Normalized cutoff frequency for the high-pass filter
sos = signal.butter(10, normalized_cutoff, btype='highpass', output='sos')  # High-pass Butterworth filter

# Function to extract features from motion data
def extract_features(signal_data):
    filtered = signal.sosfilt(sos, signal_data)
    N = len(filtered)
    fft_values = fft(filtered)
    fft_freq = fftfreq(N, 1 / sampling_rate)
    magnitude = np.abs(fft_values)
    peak_frequency = fft_freq[np.argmax(magnitude[:N // 2])]

    fab = np.abs(fft_values[:N // 2])
    F1, F2, F3, F4 = sum(np.square(fab[:25])), sum(np.square(fab[25:50])), sum(np.square(fab[50:75])), sum(np.square(fab[75:]))
    Esum = sum(np.square(fab))
    kur, ske = kurtosis(filtered, fisher=True), skew(filtered, bias=False)
    resdif, resdif2 = np.diff(filtered), np.diff(np.diff(filtered))
    Mobi = np.sqrt(np.var(resdif) / np.var(filtered))
    compx = np.sqrt(np.var(resdif2) * np.var(filtered) / (np.var(resdif) * np.var(resdif)))
    Samp, _, _ = EH.SampEn(filtered, m=2, tau=2)
    E13, E14, E15 = np.percentile(filtered, [25, 50, 75])

    return [
        np.std(filtered), np.mean(filtered), kur, ske, Mobi, compx,
        F1, F2, F3, F4, F2 / Esum, F3 / Esum, np.var(resdif2),
        Samp[0], Samp[1], Samp[2], E13, E14, E15, peak_frequency
    ]

# Function to process motion data from a given JSON file
def process_motion_file(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        print(f"Error reading file: {json_file}, Error: {e}")
        return []

    # Extract motion data from 'recording' -> 'recordedData'
    tStamp, acX, acY, acZ, agX, agY, agZ = [], [], [], [], [], [], []
    for i in data['recording']['recordedData']:
        tStamp.append(i['ts'])
        acX.append(i['data'][0])
        acY.append(i['data'][1])
        acZ.append(i['data'][2])
        agX.append(i['data'][3])
        agY.append(i['data'][4])
        agZ.append(i['data'][5])

    # Downsample by selecting actual data points (using decimation)
    downsample_factor = 4
    if len(acX) >= downsample_factor:
        tStamp = signal.decimate(tStamp, downsample_factor, zero_phase=True)
        acX = signal.decimate(acX, downsample_factor, zero_phase=True)
        acY = signal.decimate(acY, downsample_factor, zero_phase=True)
        acZ = signal.decimate(acZ, downsample_factor, zero_phase=True)
        agX = signal.decimate(agX, downsample_factor, zero_phase=True)
        agY = signal.decimate(agY, downsample_factor, zero_phase=True)
        agZ = signal.decimate(agZ, downsample_factor, zero_phase=True)

    # Feature extraction for each axis
    features_acX = extract_features(acX)
    features_acY = extract_features(acY)
    features_acZ = extract_features(acZ)
    features_agX = extract_features(agX)
    features_agY = extract_features(agY)
    features_agZ = extract_features(agZ)

    # Merge all features into a single list
    all_features = features_acX + features_acY + features_acZ + features_agX + features_agY + features_agZ

    return all_features

# Function to prepare a list of features from a given JSON file
def prepare_feature_list(json_file):
    features = process_motion_file(json_file)
    return features

#------------------------------------------------------------
# Example usage with a single JSON file
json_file_path = "./datajson/control/tremorResting_1721110619936.json"

# Prepare a list of features
feature_list = prepare_feature_list(json_file_path)

# Print the list (for verification, can be removed later)
if feature_list:
    print("\nPrepared Feature List:")
    print(feature_list)
else:
    print("No features extracted.")

# Convert to a 2D NumPy array
X_test = np.array(feature_list).reshape(1, -1)

# Predict on the test set
y_pred = model.predict(X_test)

# print output compare to target
print('predicted:',y_pred)