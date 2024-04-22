import tensorflow as tf
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the trained model
best_keras_model = load_model("best_model.h5")

# Load the new dataset
with open("./results.json", "r") as f:
    new_data = json.load(f)

# Prepare the new data
color_chars_map = {"white": 0, "black": 1, "red": 2}

# Extract the last entry of the new dataset
last_entry = new_data[-1]
last_entry_features = [
    color_chars_map[last_entry["color"]],
    last_entry["number"],
    last_entry["players"],
    last_entry["redbets"],
    last_entry["whitebets"],
    last_entry["blackbets"]
]

# Scale the features (using the same scaler as before)
scaler = StandardScaler()
last_entry_scaled = scaler.fit_transform([last_entry_features])

# Predict the next 5 results
predicted_results = []
for _ in range(5):
    # Predict the next result
    prediction = best_keras_model.predict(last_entry_scaled)

    # Convert prediction to human-readable format
    predicted_result = list(color_chars_map.keys())[np.argmax(prediction)]

    # Append predicted result to the list
    predicted_results.append(predicted_result)

    # Update last entry features with the predicted result
    last_entry_features[0] = color_chars_map[predicted_result]

    # Scale the updated features
    last_entry_scaled = scaler.transform([last_entry_features])

# Print the predicted results
print("Predicted next 5 results:", predicted_results)
