from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import time
import traceback
from datetime import datetime

# Try to import TensorFlow components
try:
    from tensorflow import keras
    import tensorflow as tf
    TF_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError as e:
    TF_AVAILABLE = False
    print(f"⚠️  TensorFlow not available: {e}")
    print("🎭 Running in demo mode")

# Try to import project modules
try:
    from config import IMG_SIZE, CLASS_NAMES
    from src.preprocess import load_and_preprocess
    CONFIG_AVAILABLE = True
    print("✅ Project config available")
except ImportError as e:
    CONFIG_AVAILABLE = False
    print(f"⚠️  Project modules not available: {e}")
    print("🔧 Using default configuration")
    IMG_SIZE = (128, 128)
    CLASS_NAMES = ["Uninfected", "Parasitized"]

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Model loading with better error handling
model = None
if TF_AVAILABLE and CONFIG_AVAILABLE:
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if os.path.exists(models_dir):
        # Look for any .keras model file
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
        if model_files:
            MODEL_PATH = os.path.join(models_dir, model_files[0])
            try:
                model = keras.models.load_model(MODEL_PATH)
                print(f"✅ Model loaded successfully: {model_files[0]}")
            except Exception as e:
                print(f"❌ Could not load model {model_files[0]}: {e}")
                print("🎭 Falling back to demo mode")
        else:
            print("⚠️  No .keras model files found in models/ directory")
            print("🎭 Running in demo mode")
    else:
        print("⚠️  Models directory not found")
        print("🎭 Running in demo mode")

def simple_preprocess(image_path, target_size):
    """Simple image preprocessing fallback with better error handling"""
    try:
        from PIL import Image
        import numpy as np
        
        # Open and convert image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        return img_array
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")

@app.route("/")
def index():
    """Landing page"""
    return render_template("landing.html")

@app.route("/diagnosis")
def diagnosis():
    """Main diagnosis / upload page"""
    return render_template("index.html")

@app.route("/research")
def research():
    """Research analysis page"""
    return render_template("research.html")

@app.route("/analytics")
def analytics():
    """Model analytics & performance dashboard"""
    return render_template("analytics.html")

@app.route("/about")
def about():
    """About page with project information"""
    try:
        return render_template("about.html")
    except Exception:
        return render_template("index.html")

@app.route("/help")
def help_page():
    """Help and FAQ page"""
    try:
        return render_template("help.html")
    except Exception:
        return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for malaria prediction with comprehensive error handling"""
    
    try:
        # Check if image was uploaded
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Validate file type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        file_ext = os.path.splitext(file.filename)[1]
        if file_ext not in allowed_extensions:
            return jsonify({"error": "Invalid file format. Please upload PNG, JPG, or JPEG files."}), 400

        # Create upload directory
        upload_dir = os.path.join(app.instance_path, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_filename = f"temp_{timestamp}_{file.filename}"
        temp_path = os.path.join(upload_dir, temp_filename)
        
        # Save file
        file.save(temp_path)
        start_time = time.time()
        
        if model is not None and TF_AVAILABLE:
            try:
                # Real model prediction
                if CONFIG_AVAILABLE:
                    img = load_and_preprocess(temp_path, target_size=IMG_SIZE)
                else:
                    img = simple_preprocess(temp_path, target_size=IMG_SIZE)
                
                # Ensure correct shape and type
                if len(img.shape) == 3:
                    img_batch = np.expand_dims(img, axis=0)
                else:
                    img_batch = img
                
                # Make prediction
                probs = model.predict(img_batch, verbose=0)[0]
                
                pred_idx = int(np.argmax(probs))
                label = CLASS_NAMES[pred_idx]
                confidence = float(probs[pred_idx])
                
                probabilities = {
                    CLASS_NAMES[0]: float(probs[0]),
                    CLASS_NAMES[1]: float(probs[1]),
                }
                
                demo_mode = False
                
            except Exception as e:
                print(f"Model prediction failed: {e}")
                print("Falling back to demo mode for this request")
                # Fall back to demo mode if model fails
                demo_mode = True
                label, confidence, probabilities = generate_demo_prediction()
                
        else:
            # Demo mode - simulate predictions
            demo_mode = True
            label, confidence, probabilities = generate_demo_prediction()

        processing_time = time.time() - start_time
        
        response_data = {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "processing_time": round(processing_time, 3),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add demo mode indicator
        if demo_mode:
            response_data["demo_mode"] = True
            
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        print(f"API Error: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500
        
    finally:
        # Clean up temp file
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Could not clean up temp file: {e}")

def generate_demo_prediction():
    """Generate realistic demo predictions"""
    import random
    time.sleep(random.uniform(0.5, 1.5))  # Simulate processing
    
    if random.random() < 0.3:  # 30% chance of parasitized
        parasitized_prob = random.uniform(0.65, 0.95)
        uninfected_prob = 1.0 - parasitized_prob
        label = "Parasitized"
        confidence = parasitized_prob
    else:
        uninfected_prob = random.uniform(0.70, 0.98)
        parasitized_prob = 1.0 - uninfected_prob
        label = "Uninfected"
        confidence = uninfected_prob
    
    probabilities = {
        "Uninfected": float(uninfected_prob),
        "Parasitized": float(parasitized_prob),
    }
    
    return label, confidence, probabilities

@app.route("/api/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "tensorflow_available": TF_AVAILABLE,
        "config_available": CONFIG_AVAILABLE,
        "demo_mode": model is None,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/analytics")
def api_analytics():
    """Return training history and evaluation metrics for the analytics dashboard"""
    import glob
    result = {}

    # Load training history (most recent file)
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    history_files = sorted(glob.glob(os.path.join(results_dir, "history_*.json")), reverse=True)
    if history_files:
        try:
            import json
            with open(history_files[0]) as f:
                history = json.load(f)
            result["history"] = history
            result["epochs"] = list(range(1, len(history.get("accuracy", [])) + 1))
        except Exception as e:
            result["history_error"] = str(e)

    # Load evaluation report
    eval_path = os.path.join(results_dir, "evaluation_report.json")
    if os.path.exists(eval_path):
        try:
            import json
            with open(eval_path) as f:
                evaluation = json.load(f)
            result["evaluation"] = evaluation
        except Exception as e:
            result["evaluation_error"] = str(e)

    result["model_loaded"] = model is not None
    result["timestamp"] = datetime.now().isoformat()
    return jsonify(result)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File too large. Maximum size is 10MB."}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template("index.html"), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    print(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error occurred"}), 500

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 ParaDetect AI - Malaria Detection System")
    print("=" * 60)
    print(f"📊 Model status: {'✅ Loaded' if model else '🎭 Demo Mode'}")
    print(f"🧠 TensorFlow: {'✅ Available' if TF_AVAILABLE else '❌ Not Available'}")
    print(f"⚙️  Config: {'✅ Available' if CONFIG_AVAILABLE else '❌ Using Defaults'}")
    print(f"🌐 Server starting at: http://localhost:5000")
    print("📱 Open your browser and navigate to the URL above")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)