import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, 
    precision_score, recall_score, f1_score, roc_curve, auc
)

# Define the file path for the model and the test features CSV
from config import base_path
modeltouse = "trained_model_pinch_fe_SMOTE.pkl"
featuretotest = "pinch_featureAppdata.csv"

new_file_path_modeltouse = os.path.join(base_path, modeltouse)
new_file_path_featuretotest = os.path.join(base_path, featuretotest)

# Load the CSV file with UTF-8 encoding
df = pd.read_csv(new_file_path_featuretotest, encoding='utf-8')

# Filter out the specified columns
columns_to_remove = ['Index', 'Folder Name', 'Timestamp Year', 'Age', 'Gender', 'Birthdate Year']
filtered_df = df.drop(columns=columns_to_remove)

# Filter to keep only 'pd' and 'ct' groups in the 'Diagnosis' column
filtered_df = filtered_df[filtered_df['Diagnosis'].isin(['pd', 'ct'])]

# Select only the columns starting from 'mdia_std' to the end for features
features = filtered_df.loc[:, 'mdia_std':]

# Target variable
target = filtered_df['Diagnosis']

# Encode 'Diagnosis' as binary classification (0 for 'ct' and 1 for 'pd')
y_true = np.where(target == 'pd', 1, 0)

# Prepare features
X_test = features.values

# Load the trained model from the pickle file
with open(new_file_path_modeltouse, 'rb') as f:
    model = pickle.load(f)

# Predict on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # For ROC and AUC calculation

# print output compare to target
print('target:',y_true)
print('predicted:',y_pred)

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Detailed classification report
report = classification_report(y_true, y_pred, target_names=['ct', 'pd'])

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc_value = auc(fpr, tpr)

# Display performance metrics
print("Model Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:")
print(report)

# Optionally plot the ROC curve
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_value:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
