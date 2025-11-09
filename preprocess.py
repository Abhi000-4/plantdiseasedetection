import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd

# Extract color histogram features
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))  # Resize for simplicity
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()  # 512 features

# Load dataset
data_dir = r'C:\Users\abhia\Downloads\soyabean'  # Replace with your path
features = []
labels = []
class_map = {}  # Map class names to IDs
class_id = 0

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        class_map[class_id] = class_name
        for img_file in os.listdir(class_path):
            path = os.path.join(class_path, img_file)
            feat = extract_features(path)
            if feat is not None:
                features.append(feat)
                labels.append(class_id)
        class_id += 1

# Save class map
pd.DataFrame(list(class_map.items()), columns=['ID', 'Class']).to_csv('class_map.csv', index=False)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("Preprocessing complete. Classes:", class_map)