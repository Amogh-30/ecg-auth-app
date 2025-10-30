import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import os

print("--- Phase 1: Enrollment Database Creation ---")

# --- Step 1: Define Constants ---
SEED = 42
TEST_SIZE = 0.20
NUM_USERS = 90
FEATURE_DIM = 96

# --- Step 2: Load Required Files ---
try:
    X_all = np.load("X_all.npy")
    y_all = np.load("y_all.npy")
    X_train_features_rp = np.load("X_train_rp_all.npy")
    
    # --- CHANGE: Load .keras model ---
    feature_model = load_model("feature_model_all.keras")
    rp = joblib.load("rp_transformer_all.joblib")
    scaler = joblib.load("scaler_all.joblib")
    # ---------------------------------
    
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    print("Please make sure all model and data .npy/.joblib/.keras files are in the same folder.")
    exit()

print(f"Loaded {len(X_all)} total samples and {len(X_train_features_rp)} training features.")

# --- Step 3: Re-create the Training Labels ---
_, _, y_train_labels, _ = train_test_split(
    y_all, y_all,
    test_size=TEST_SIZE, 
    stratify=y_all, 
    random_state=SEED
)

if len(y_train_labels) != len(X_train_features_rp):
    print(f"Error: Mismatch in data length!")
    exit()

print(f"Successfully recreated {len(y_train_labels)} training labels.")

# --- Step 4: Calculate and Store Master Templates ---
template_database = {} 
for user_id in range(NUM_USERS):
    user_indices = np.where(y_train_labels == user_id)[0]
    if len(user_indices) == 0:
        continue
    user_features = X_train_features_rp[user_indices]
    master_template = np.mean(user_features, axis=0)
    username = f"Person_{user_id + 1:02d}"
    template_database[username] = master_template

print(f"Created {len(template_database)} master templates.")

# --- Step 5: Save the Enrollment Database ---
db_filename = "user_template_database.npy"
np.save(db_filename, template_database)

print(f"--- Enrollment Complete! Database saved to '{db_filename}' ---")