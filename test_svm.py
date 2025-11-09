import cv2
import numpy as np
import joblib
import pandas as pd

svm = joblib.load('svm_model.pkl')
class_map = pd.read_csv('class_map.csv').set_index('ID')['Class'].to_dict()

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten().reshape(1, -1)

# Upload a test image (manually place in folder or use file dialog)
image_path = 'C:\\Users\\abhia\\Downloads\\download.jpg'  # Replace with a test image path
features = extract_features(image_path)
if features is not None:
    prediction = svm.predict(features)[0]
    disease = class_map[prediction]
    print(f"Predicted: {disease}")
else:
    print("Failed to load test image")