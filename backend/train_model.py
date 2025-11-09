import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATASET_PATH = r"C:\Users\abhia\Downloads\soyabean"
IMAGE_SIZE = (128, 128)
TEST_SIZE = 0.2
RANDOM_STATE = 42

print("="*70)
print("🌱 SOYBEAN DISEASE DETECTION - MODEL TRAINING")
print("="*70)

def extract_features(image_path):
    """
    Extract features from image for SVM training.
    Must match the feature extraction in app.py!
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return None
        
        # Resize to standard size
        img_resized = cv2.resize(img, IMAGE_SIZE)
        
        # METHOD 1: Color Histograms (More robust for disease detection)
        # Convert to different color spaces
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        
        # Calculate histograms
        # HSV histograms
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # LAB histograms
        hist_l = cv2.calcHist([lab], [0], None, [32], [0, 256])
        hist_a = cv2.calcHist([lab], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([lab], [2], None, [32], [0, 256])
        
        # BGR histograms
        hist_b_bgr = cv2.calcHist([img_resized], [0], None, [32], [0, 256])
        hist_g_bgr = cv2.calcHist([img_resized], [1], None, [32], [0, 256])
        hist_r_bgr = cv2.calcHist([img_resized], [2], None, [32], [0, 256])
        
        # Texture features (mean and std of each channel)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        texture_features = np.array([
            np.mean(gray), np.std(gray),
            np.mean(hsv[:,:,0]), np.std(hsv[:,:,0]),
            np.mean(hsv[:,:,1]), np.std(hsv[:,:,1]),
            np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])
        ])
        
        # Concatenate all features
        features = np.concatenate([
            hist_h.flatten(),
            hist_s.flatten(),
            hist_v.flatten(),
            hist_l.flatten(),
            hist_a.flatten(),
            hist_b.flatten(),
            hist_b_bgr.flatten(),
            hist_g_bgr.flatten(),
            hist_r_bgr.flatten(),
            texture_features
        ])
        
        # Normalize
        features = features / (features.sum() + 1e-7)
        
        return features
    
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

def load_dataset(dataset_path):
    """Load images and labels from dataset"""
    print(f"\n📂 Loading dataset from: {dataset_path}")
    
    X = []  # Features
    y = []  # Labels
    class_names = []
    
    # Get all class folders
    classes = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"📋 Found {len(classes)} classes: {classes}")
    
    for class_idx, class_name in enumerate(sorted(classes)):
        class_path = os.path.join(dataset_path, class_name)
        class_names.append(class_name)
        
        # Get all images in this class
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n📁 Processing class: {class_name} ({len(image_files)} images)")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            features = extract_features(img_path)
            
            if features is not None:
                X.append(features)
                y.append(class_idx)
        
        print(f"   ✅ Loaded {len([label for label in y if label == class_idx])} samples")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   • Total samples: {len(X)}")
    print(f"   • Feature dimensions: {X.shape[1]}")
    print(f"   • Classes: {len(class_names)}")
    
    return X, y, class_names

def train_model(X, y, class_names):
    """Train SVM model"""
    print("\n" + "="*70)
    print("🤖 TRAINING MODEL")
    print("="*70)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"   • Training samples: {len(X_train)}")
    print(f"   • Testing samples: {len(X_test)}")
    
    # Scale features
    print(f"\n⚙️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    print(f"\n🔧 Training SVM classifier...")
    print(f"   Kernel: RBF")
    print(f"   This may take a few minutes...")
    
    model = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,  # Enable probability estimates
        random_state=RANDOM_STATE
    )
    
    model.fit(X_train_scaled, y_train)
    print(f"   ✅ Training complete!")
    
    # Evaluate
    print(f"\n📈 Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print(f"🎯 MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy*100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("✅ Confusion matrix saved as 'confusion_matrix.png'")
    
    return model, scaler, accuracy

def save_model(model, scaler, class_names, feature_dim):
    """Save model and related files"""
    print(f"\n💾 Saving model and related files...")
    
    # Save model
    joblib.dump(model, 'svm_model.pkl')
    print(f"   ✅ Model saved: svm_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    print(f"   ✅ Scaler saved: scaler.pkl")
    
    # Save class map
    class_map = pd.DataFrame({
        'class_id': range(len(class_names)),
        'class_name': class_names
    })
    class_map.to_csv('class_map.csv', index=False)
    print(f"   ✅ Class map saved: class_map.csv")
    print(f"\n   Class mapping:")
    for idx, name in enumerate(class_names):
        print(f"      {idx} -> {name}")
    
    # Save training info
    training_info = {
        'feature_dimensions': feature_dim,
        'num_classes': len(class_names),
        'classes': class_names,
        'image_size': IMAGE_SIZE,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE
    }
    
    with open('training_info.json', 'w') as f:
        import json
        json.dump(training_info, f, indent=4)
    print(f"   ✅ Training info saved: training_info.json")

def test_single_prediction(model, scaler, class_names, test_image_path):
    """Test prediction on a single image"""
    print(f"\n🧪 Testing single prediction...")
    print(f"   Image: {test_image_path}")
    
    if not os.path.exists(test_image_path):
        print(f"   ❌ Image not found!")
        return
    
    features = extract_features(test_image_path)
    if features is None:
        print(f"   ❌ Failed to extract features!")
        return
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    print(f"\n   Prediction: {class_names[prediction]}")
    print(f"   Confidence: {probabilities[prediction]*100:.2f}%")
    print(f"\n   All probabilities:")
    for idx, prob in enumerate(probabilities):
        print(f"      {class_names[idx]}: {prob*100:.2f}%")

if __name__ == "__main__":
    try:
        # Check if dataset exists
        if not os.path.exists(DATASET_PATH):
            print(f"\n❌ Dataset path not found: {DATASET_PATH}")
            print(f"   Please update DATASET_PATH in the script")
            exit(1)
        
        # Load dataset
        X, y, class_names = load_dataset(DATASET_PATH)
        
        if len(X) == 0:
            print(f"\n❌ No images found in dataset!")
            exit(1)
        
        # Train model
        model, scaler, accuracy = train_model(X, y, class_names)
        
        # Save everything
        save_model(model, scaler, class_names, X.shape[1])
        
        # Test on a sample image
        # Update this path to test with your own image
        sample_image = os.path.join(DATASET_PATH, class_names[0], 
                                   os.listdir(os.path.join(DATASET_PATH, class_names[0]))[0])
        test_single_prediction(model, scaler, class_names, sample_image)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\nFiles created:")
        print(f"   • svm_model.pkl")
        print(f"   • scaler.pkl")
        print(f"   • class_map.csv")
        print(f"   • training_info.json")
        print(f"   • confusion_matrix.png")
        print(f"\nModel accuracy: {accuracy*100:.2f}%")
        print(f"\n💡 Next steps:")
        print(f"   1. Copy the files to your backend folder")
        print(f"   2. Update app.py to use the scaler")
        print(f"   3. Restart the Flask server")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()