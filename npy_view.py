import numpy as np
import matplotlib.pyplot as plt
# Load the database file
# allow_pickle=True is required to load a dictionary
db = np.load("user_template_database.npy", allow_pickle=True).item()

# --- Now you can inspect it ---

# Print the username for the first user
print("--- Username for User 1 ---")
print(list(db.keys())[0])  # Output: 'Person_01'

# Print the 96-dim master template vector for Person_01
print("\n--- Template for Person_01 (first 5 values) ---")
print(db['Person_01'])

# Print the shape of the template
print("\n--- Shape of Template ---")
print(db['Person_01'].shape) # Output: (96,)
# Load the database
db = np.load("user_template_database.npy", allow_pickle=True).item()
template = db["Person_01"]
# # Load a "live" signal
# signal = np.load("simulated_login_attempts/Person_01/attempt_0001.npy")

# plt.plot(signal)
# plt.title("Original Raw ECG Signal (1500 points)")
# plt.show()
#Dont use the below stuff
# plt.bar(range(len(template)), template)
# plt.title("Master Template for Person_01 (96 features)")
# plt.show()