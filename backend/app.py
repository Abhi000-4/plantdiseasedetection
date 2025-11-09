from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename
import traceback
from blockchain import Blockchain

app = Flask(__name__)

# Enable CORS for all routes - IMPORTANT for frontend to connect
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = (128, 128)  # Must match training

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model, scaler, class map and blockchain
model = None
scaler = None
class_map_df = None
blockchain = None

def load_model():
    """Load the trained model, scaler and class map"""
    global model, scaler, class_map_df
    
    try:
        print("📂 Current working directory:", os.getcwd())
        print("📂 Files in directory:", os.listdir('.'))
        print()
        
        # Try to load model
        model_path = 'svm_model.pkl'
        if os.path.exists(model_path):
            print(f"📂 Loading model from: {model_path}")
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully: {type(model)}")
            
            # Try to get the number of features the model expects
            if hasattr(model, 'n_features_in_'):
                print(f"   Model expects {model.n_features_in_} features")
        else:
            print(f"❌ Model file not found: {model_path}")
            print("   Please run train_model.py first to train the model")
            model = None
        
        # Try to load scaler
        scaler_path = 'scaler.pkl'
        if os.path.exists(scaler_path):
            print(f"📂 Loading scaler from: {scaler_path}")
            scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded successfully")
        else:
            print(f"⚠️  Scaler file not found: {scaler_path}")
            print("   Model may not work correctly without scaler")
            print("   Please run train_model.py to generate scaler.pkl")
            scaler = None
        
        # Try to load class map
        classmap_path = 'class_map.csv'
        if os.path.exists(classmap_path):
            print(f"📂 Loading class map from: {classmap_path}")
            class_map_df = pd.read_csv(classmap_path)
            print(f"✅ Class map loaded: {len(class_map_df)} classes")
            
            # Show class map structure
            print(f"   Columns: {class_map_df.columns.tolist()}")
            if 'class_name' in class_map_df.columns:
                print(f"   Classes: {class_map_df['class_name'].tolist()}")
            else:
                print(f"   First few rows:\n{class_map_df.head()}")
        else:
            print(f"❌ Class map file not found: {classmap_path}")
            print("   Please run train_model.py to generate class_map.csv")
            class_map_df = None
        
        print()
        return model is not None and class_map_df is not None
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return False

def init_blockchain():
    """Initialize the blockchain"""
    global blockchain
    try:
        print("🔗 Initializing blockchain...")
        blockchain = Blockchain(storage_file='predictions_blockchain.json')
        print(f"✅ Blockchain ready: {len(blockchain.chain)} blocks")
        
        # Verify blockchain integrity
        if blockchain.is_chain_valid():
            print("✅ Blockchain integrity verified")
        else:
            print("⚠️  Blockchain integrity check failed!")
        
        print()
        return True
    except Exception as e:
        print(f"❌ Error initializing blockchain: {e}")
        traceback.print_exc()
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image):
    """
    Extract features from image for prediction.
    MUST MATCH the feature extraction used during training!
    """
    try:
        # Resize to standard size (same as training)
        img_resized = cv2.resize(image, IMAGE_SIZE)
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        
        # Calculate color histograms (32 bins each)
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
        
        # Texture features (mean and standard deviation)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        texture_features = np.array([
            np.mean(gray), np.std(gray),
            np.mean(hsv[:,:,0]), np.std(hsv[:,:,0]),
            np.mean(hsv[:,:,1]), np.std(hsv[:,:,1]),
            np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])
        ])
        
        # Concatenate all features
        features = np.concatenate([
            hist_h.flatten(),      # 32 features
            hist_s.flatten(),      # 32 features
            hist_v.flatten(),      # 32 features
            hist_l.flatten(),      # 32 features
            hist_a.flatten(),      # 32 features
            hist_b.flatten(),      # 32 features
            hist_b_bgr.flatten(),  # 32 features
            hist_g_bgr.flatten(),  # 32 features
            hist_r_bgr.flatten(),  # 32 features
            texture_features       # 8 features
        ])  # Total: 296 features
        
        # Normalize
        features = features / (features.sum() + 1e-7)
        
        print(f"   ✓ Features extracted: shape={features.shape}, expected=(296,)")
        
        # Verify feature count
        if features.shape[0] != 296:
            raise ValueError(f"Feature extraction produced {features.shape[0]} features, expected 296")
        
        return features.reshape(1, -1)
        
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        traceback.print_exc()
        raise

def calculate_moisture(image):
    """Calculate moisture level from image"""
    try:
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract saturation channel (moisture indicator)
        saturation = hsv[:, :, 1]
        
        # Calculate average saturation as moisture proxy
        moisture_level = int(np.mean(saturation) / 2.55)  # Convert to 0-100 scale
        
        # Ensure it's within 0-100 range
        moisture_level = max(0, min(100, moisture_level))
        
        return moisture_level
    except Exception as e:
        print(f"❌ Error calculating moisture: {e}")
        return 50  # Return default value

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    model_info = {}
    if model is not None:
        if hasattr(model, 'n_features_in_'):
            model_info['expected_features'] = int(model.n_features_in_)
        model_info['scaler_loaded'] = scaler is not None
    
    blockchain_info = {}
    if blockchain is not None:
        blockchain_info = {
            'blocks': len(blockchain.chain),
            'valid': blockchain.is_chain_valid(),
            'predictions_count': len(blockchain.get_predictions_history())
        }
    
    return jsonify({
        'status': 'running',
        'message': 'Plant Disease Detection API with Blockchain',
        'version': '2.0',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'blockchain_enabled': blockchain is not None,
        'model_info': model_info,
        'blockchain_info': blockchain_info,
        'endpoints': {
            '/': 'GET - API status',
            '/predict': 'POST - Upload image for disease prediction',
            '/blockchain': 'GET - View entire blockchain',
            '/blockchain/history': 'GET - View predictions history',
            '/blockchain/verify': 'GET - Verify blockchain integrity',
            '/blockchain/stats': 'GET - View blockchain statistics',
            '/blockchain/block/<index>': 'GET - View specific block'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image"""
    try:
        print("\n" + "="*60)
        print("📥 NEW PREDICTION REQUEST")
        print("="*60)
        
        # Check if file is in request
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({'error': 'No file part in request'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            print("❌ No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"📁 File received: {file.filename}")
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            print(f"❌ Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
        
        # Read and decode image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("❌ Could not decode image")
            return jsonify({'error': 'Could not read image. File may be corrupted'}), 400
        
        print(f"✅ Image loaded: shape={img.shape}, dtype={img.dtype}")
        
        # Calculate moisture level (works without model)
        print("💧 Calculating moisture level...")
        moisture = calculate_moisture(img)
        print(f"✅ Moisture: {moisture}%")
        
        # Check if model is loaded
        if model is None or class_map_df is None:
            print("⚠️  Model not loaded - returning test response")
            response = {
                'disease': 'model_not_loaded',
                'moisture': moisture,
                'status': 'error',
                'message': 'Model not loaded. Please run train_model.py first.'
            }
            print(f"📤 Response: {response}")
            print("="*60 + "\n")
            return jsonify(response), 200
        
        # Extract features for prediction
        print("🔬 Extracting features...")
        features = extract_features(img)
        print(f"✅ Feature extraction successful: {features.shape}")
        
        # Scale features if scaler is available
        if scaler is not None:
            print("⚖️  Scaling features...")
            features_scaled = scaler.transform(features)
            print(f"✅ Features scaled")
        else:
            print("⚠️  No scaler available - using unscaled features")
            print("   Predictions may be inaccurate!")
            features_scaled = features
        
        # Make prediction
        print("🤖 Making prediction...")
        prediction = model.predict(features_scaled)[0]
        print(f"✅ Raw prediction: {prediction} (type: {type(prediction)})")
        
        # Get disease name from class map
        disease = None
        
        # Try different column names for class identification
        if 'class_id' in class_map_df.columns and 'class_name' in class_map_df.columns:
            disease_rows = class_map_df[class_map_df['class_id'] == prediction]
            if len(disease_rows) > 0:
                disease = disease_rows['class_name'].values[0]
        elif 'label' in class_map_df.columns:
            disease_rows = class_map_df[class_map_df.index == prediction]
            if len(disease_rows) > 0:
                disease = disease_rows['label'].values[0]
        
        # Fallback: use prediction as index
        if disease is None:
            if prediction < len(class_map_df):
                # Assume first text column contains class names
                text_columns = class_map_df.select_dtypes(include=['object']).columns
                if len(text_columns) > 0:
                    disease = class_map_df.iloc[prediction][text_columns[0]]
                else:
                    disease = f"class_{prediction}"
            else:
                disease = f"unknown_class_{prediction}"
        
        print(f"✅ Disease identified: {disease}")
        
        # Get confidence score if available
        confidence = 0.85  # Default
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(features_scaled)[0]
                confidence = float(np.max(proba))
                print(f"✅ Confidence: {confidence:.2%}")
                
                # Show all class probabilities
                print(f"   All probabilities:")
                for idx, prob in enumerate(proba):
                    if idx < len(class_map_df):
                        class_name = class_map_df.iloc[idx]['class_name'] if 'class_name' in class_map_df.columns else f"class_{idx}"
                        print(f"      {class_name}: {prob:.2%}")
            except Exception as prob_error:
                print(f"⚠️  Could not get probabilities: {prob_error}")
        
        # Prepare response
        response = {
            'disease': str(disease),
            'moisture': int(moisture),
            'confidence': round(float(confidence), 2),
            'status': 'success'
        }
        
        # Add prediction to blockchain
        if blockchain is not None:
            try:
                print("🔗 Recording prediction on blockchain...")
                prediction_data = {
                    'disease': str(disease),
                    'moisture': int(moisture),
                    'confidence': round(float(confidence), 2),
                    'image_name': file.filename
                }
                block = blockchain.add_prediction_block(prediction_data)
                response['blockchain'] = {
                    'recorded': True,
                    'block_index': block.index,
                    'block_hash': block.hash,
                    'timestamp': str(block.timestamp)
                }
                print(f"✅ Prediction recorded on blockchain at block {block.index}")
                print(f"   Block hash: {block.hash}")
            except Exception as bc_error:
                print(f"⚠️  Failed to add to blockchain: {bc_error}")
                traceback.print_exc()
                response['blockchain'] = {
                    'recorded': False,
                    'error': str(bc_error)
                }
        else:
            response['blockchain'] = {
                'recorded': False,
                'error': 'Blockchain not initialized'
            }
        
        print(f"📤 Response: {response}")
        print("="*60 + "\n")
        return jsonify(response), 200
        
    except Exception as e:
        print(f"\n❌ ERROR IN PREDICTION:")
        print(f"   {str(e)}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        print("="*60 + "\n")
        
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/blockchain', methods=['GET'])
def get_blockchain():
    """Get the entire blockchain"""
    try:
        if blockchain is None:
            return jsonify({'error': 'Blockchain not initialized'}), 500
        
        chain_data = blockchain.get_chain_data()
        return jsonify({
            'chain': chain_data,
            'length': len(chain_data),
            'valid': blockchain.is_chain_valid(),
            'message': 'Blockchain retrieved successfully'
        }), 200
    except Exception as e:
        print(f"❌ Error getting blockchain: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/blockchain/history', methods=['GET'])
def get_predictions_history():
    """Get all predictions from blockchain"""
    try:
        if blockchain is None:
            return jsonify({'error': 'Blockchain not initialized'}), 500
        
        history = blockchain.get_predictions_history()
        return jsonify({
            'predictions': history,
            'count': len(history),
            'message': 'Predictions history retrieved successfully'
        }), 200
    except Exception as e:
        print(f"❌ Error getting predictions history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/blockchain/verify', methods=['GET'])
def verify_blockchain():
    """Verify blockchain integrity"""
    try:
        if blockchain is None:
            return jsonify({'error': 'Blockchain not initialized'}), 500
        
        is_valid = blockchain.is_chain_valid()
        return jsonify({
            'valid': is_valid,
            'blocks': len(blockchain.chain),
            'predictions': len(blockchain.get_predictions_history()),
            'message': 'Blockchain is valid and has not been tampered with ✅' if is_valid else 'Blockchain has been compromised! ❌'
        }), 200
    except Exception as e:
        print(f"❌ Error verifying blockchain: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/blockchain/block/<int:index>', methods=['GET'])
def get_block(index):
    """Get a specific block by index"""
    try:
        if blockchain is None:
            return jsonify({'error': 'Blockchain not initialized'}), 500
        
        block = blockchain.get_block_by_index(index)
        if block:
            return jsonify({
                'block': block,
                'message': f'Block {index} retrieved successfully'
            }), 200
        else:
            return jsonify({'error': f'Block {index} not found'}), 404
    except Exception as e:
        print(f"❌ Error getting block: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/blockchain/stats', methods=['GET'])
def get_blockchain_stats():
    """Get blockchain statistics"""
    try:
        if blockchain is None:
            return jsonify({'error': 'Blockchain not initialized'}), 500
        
        predictions = blockchain.get_predictions_history()
        
        # Calculate statistics
        disease_counts = {}
        for pred in predictions:
            disease = pred.get('disease', 'unknown')
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        avg_moisture = sum(p.get('moisture', 0) for p in predictions) / len(predictions) if predictions else 0
        avg_confidence = sum(p.get('confidence', 0) for p in predictions) / len(predictions) if predictions else 0
        
        return jsonify({
            'total_blocks': len(blockchain.chain),
            'total_predictions': len(predictions),
            'disease_distribution': disease_counts,
            'average_moisture': round(avg_moisture, 2),
            'average_confidence': round(avg_confidence, 2),
            'blockchain_valid': blockchain.is_chain_valid(),
            'genesis_block_hash': blockchain.chain[0].hash if blockchain.chain else None
        }), 200
    except Exception as e:
        print(f"❌ Error getting blockchain stats: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌱 PLANT DISEASE DETECTION WITH BLOCKCHAIN v2.0")
    print("="*70)
    
    # Load model
    print("\n🔧 Initializing AI Model...")
    model_loaded = load_model()
    
    # Initialize blockchain
    print("🔗 Initializing Blockchain System...")
    blockchain_loaded = init_blockchain()
    
    print("\n📊 System Status:")
    print(f"   • AI Model loaded: {'✅ YES' if model is not None else '❌ NO'}")
    print(f"   • Scaler loaded: {'✅ YES' if scaler is not None else '❌ NO'}")
    print(f"   • Class map loaded: {'✅ YES' if class_map_df is not None else '❌ NO'}")
    
    if model is not None and hasattr(model, 'n_features_in_'):
        print(f"   • Model expects: {model.n_features_in_} features")
    
    print(f"   • Blockchain initialized: {'✅ YES' if blockchain is not None else '❌ NO'}")
    
    if blockchain is not None:
        print(f"   • Blockchain blocks: {len(blockchain.chain)}")
        print(f"   • Recorded predictions: {len(blockchain.get_predictions_history())}")
    
    print(f"   • Upload folder: {UPLOAD_FOLDER}")
    print(f"   • CORS enabled: ✅ YES (http://localhost:3000)")
    
    if not model_loaded:
        print("\n⚠️  WARNING: Model not loaded properly")
        print("   Please run train_model.py to train and generate:")
        print("   • svm_model.pkl")
        print("   • scaler.pkl")
        print("   • class_map.csv")
        print(f"   Current directory: {os.getcwd()}")
    
    if scaler is None and model is not None:
        print("\n⚠️  WARNING: Scaler not loaded")
        print("   Predictions may be inaccurate without the scaler")
        print("   Please run train_model.py to generate scaler.pkl")
    
    print("\n" + "="*70)
    print("🚀 Starting Server...")
    print("="*70)
    print(f"📡 Local:   http://127.0.0.1:5000")
    print(f"📡 Network: http://0.0.0.0:5000")
    print("\n📋 Available Endpoints:")
    print("   • /predict - Upload image for prediction")
    print("   • /blockchain - View entire blockchain")
    print("   • /blockchain/history - View predictions history")
    print("   • /blockchain/verify - Verify blockchain integrity")
    print("   • /blockchain/stats - View blockchain statistics")
    print("="*70)
    print("\n💡 Press CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)