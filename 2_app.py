import numpy as np
import os
import joblib
# REMOVED: from tensorflow.keras.models import load_model
import onnxruntime as ort # ADDED
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 

# --- 1. Initialize Flask App ---
app = Flask(__name__, static_folder='static', template_folder='.')
CORS(app) 

# --- 2. Define Constants ---
AUTH_THRESHOLD = 0.80 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DATA_DIR = os.path.join(BASE_DIR, "simulated_login_attempts")

# --- 3. Load All Models and Database (Done ONCE at startup) ---
print("Loading all models, this may take a moment...")
try:
    # --- CHANGED: Load ONNX model ---
    onnx_path = os.path.join(BASE_DIR, "feature_model.onnx")
    ort_session = ort.InferenceSession(onnx_path)
    # -------------------------------
    
    rp = joblib.load(os.path.join(BASE_DIR, "rp_transformer_all.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler_all.joblib"))
    template_database = np.load(os.path.join(BASE_DIR, "user_template_database.npy"), allow_pickle=True).item()
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find model files: {e}")
    print("Please make sure all .onnx, .joblib, and .npy files are in the same folder.")
    print("AND ensure you have run '1_create_enrollment_db.py' first.")
    exit()

print(f"Models and database with {len(template_database)} users loaded.")

# --- 4. Helper Function: Biometric Processing Pipeline (Now using ONNX) ---
def process_live_signal(signal_array):
    """Converts a raw 1500-sample signal into a 96-dim template."""
    std_dev = np.std(signal_array)
    if std_dev == 0: std_dev = 1.0
    signal_norm = (signal_array - np.mean(signal_array)) / std_dev
    
    # 2. Reshape for ONNX (batch_size=1, steps=1500, channels=1)
    #    Make sure it's float32, as required by the model
    signal_onnx = signal_norm.reshape(1, 1500, 1).astype(np.float32)
    
    # 3. Process through the full pipeline
    # --- CHANGED: Use onnxruntime ---
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    features_128d = ort_session.run([output_name], {input_name: signal_onnx})[0]
    # -------------------------------
    
    features_96d = rp.transform(features_128d)
    features_96d_scaled = scaler.transform(features_96d)
    
    return features_96d_scaled

# --- 5. API Endpoint: Get List of Test Files ---
@app.route("/get_test_files")
def get_test_files():
    file_map = {}
    try:
        for user_dir in os.listdir(SIM_DATA_DIR):
            user_path = os.path.join(SIM_DATA_DIR, user_dir)
            if os.path.isdir(user_path):
                files = [f for f in os.listdir(user_path) if f.endswith('.npy')]
                file_map[user_dir] = files
        return jsonify(file_map)
    except FileNotFoundError:
        return jsonify({"error": "Simulation data not found on server."}), 404

# --- 6. API Endpoint: Authentication Logic ---
@app.route("/authenticate", methods=["POST"])
def authenticate_user():
    data = request.json
    username_claim = data.get("username")   
    filename_attempt = data.get("filename") 

    if not username_claim or not filename_attempt:
        return jsonify({"error": "Missing username or filename"}), 400
    if username_claim not in template_database:
        return jsonify({"error": "User not enrolled"}), 404
        
    try:
        file_path = os.path.join(SIM_DATA_DIR, username_claim, filename_attempt)
        if not os.path.normpath(file_path).startswith(os.path.normpath(SIM_DATA_DIR)):
             return jsonify({"error": "Invalid file path"}), 400
        live_signal_array = np.load(file_path)
    except FileNotFoundError:
        return jsonify({"authenticated": False, "message": "Test file not found. This might be an impostor attempt."}), 404
    except Exception as e:
        return jsonify({"error": f"Error loading file: {str(e)}"}), 500

    stored_template = template_database[username_claim].reshape(1, -1)
    live_template = process_live_signal(live_signal_array)
    similarity = cosine_similarity(live_template, stored_template)[0][0]
    
    if similarity >= AUTH_THRESHOLD:
        return jsonify({
            "authenticated": True, "message": f"Welcome, {username_claim}!",
            "user": username_claim, "similarity": float(similarity)
        })
    else:
        return jsonify({
            "authenticated": False, "message": f"Access Denied. You are not {username_claim}.",
            "user": username_claim, "similarity": float(similarity)
        }), 401

# --- 7. API Endpoint: Serve the Frontend HTML ---
@app.route("/")
def index():
    return render_template("index.html") # (Assuming you renamed 3_index.html)

# --- 8. Run the Server ---
if __name__ == "__main__":
    print("Flask server running! Open http://1227.0.0.1:5000 in your browser.")
    app.run(debug=True) # You can add threaded=True back if you like