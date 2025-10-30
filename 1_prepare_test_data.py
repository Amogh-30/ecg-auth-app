import numpy as np
import os
from sklearn.model_selection import train_test_split

print("--- Preparing Simulated Login Attempt Files ---")

# --- Define Constants (MUST match your notebook) ---
SEED = 42
TEST_SIZE = 0.20
OUTPUT_DIR = "simulated_login_attempts"

# --- Load Original Data ---
try:
    X_all = np.load("X_all.npy")
    y_all = np.load("y_all.npy")
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    print("Please make sure 'X_all.npy' and 'y_all.npy' are in this folder.")
    exit()

# --- Re-create the Test Set ---
# We split the *raw* X_all and y_all to get the *raw* X_test signals
# that your model has never seen.
_, X_test_raw, _, y_test_labels = train_test_split(
    X_all, 
    y_all, 
    test_size=TEST_SIZE, 
    stratify=y_all, 
    random_state=SEED
)

print(f"Loaded {len(X_test_raw)} raw test signals for simulation.")

# --- Save Each Test Signal as a Separate .npy File ---
if os.path.exists(OUTPUT_DIR):
    import shutil
    shutil.rmtree(OUTPUT_DIR) # Clear old attempts
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(len(X_test_raw)):
    signal_array = X_test_raw[i].squeeze() # Get the (1500,) array
    user_id = y_test_labels[i]
    username = f"Person_{user_id + 1:02d}" # e.g., "Person_01"
    
    # Create a subfolder for this user
    user_dir = os.path.join(OUTPUT_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    
    # Save the signal as a "live attempt" file
    output_path = os.path.join(user_dir, f"attempt_{i:04d}.npy")
    np.save(output_path, signal_array)

print(f"Done. Created '{OUTPUT_DIR}' folder with {len(X_test_raw)} simulated .npy login files.")