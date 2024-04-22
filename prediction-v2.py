import tensorflow as tf
import numpy as np
import pandas as pd
import json

history_file = open("../../datasets/history-250K.json", "r")
json_history = json.load(history_file)

results = []

for i in range(len(json_history)):
    color_char_map = {"white": "B", "black": "P", "red": "V"}
    results.append(color_char_map[json_history[i]["color"]])

# Check if there are at least 20 results
if len(results) < 20:
    print("There are not enough results to make a prediction.")
    exit()

# Define a dictionary to encode the colors as 0 (for black or white) and 1 (for red)
colors = {"V": 1, "P": 0, "B": 0}

# Create a pandas DataFrame with the previous roulette results and the corresponding encoded colors
df = pd.DataFrame({"result": results, "color": [colors[r] for r in results]})

# Split the DataFrame into training and test sets
train_df = df.iloc[:15]
test_df = df.iloc[15:]

# Create a neural network model using TensorFlow
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(8, activation="sigmoid", input_shape=(1,)),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]
)

# Compile the model and define the hyperparameters
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(
    train_df["color"],
    tf.keras.utils.to_categorical(train_df.index % 3, num_classes=3),
    epochs=1000,
    verbose=0,
)

# Use the trained model to predict the next color of the roulette
last_result = test_df.iloc[-1]["color"]
predictions = model.predict(np.array([last_result]))
next_result = np.argmax(predictions)
next_result_prob = predictions[0][next_result]

# Print the prediction for the next color of the roulette, including the probability
if next_result == 0:
    print(f"Next result: black, probability: {next_result_prob:.2f}")
elif next_result == 1:
    print(f"Next result: red, probability: {next_result_prob:.2f}")
else:
    print(f"Next result: white, probability: {next_result_prob:.2f}")
