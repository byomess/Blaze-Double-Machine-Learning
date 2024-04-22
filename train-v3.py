import tensorflow as tf
import numpy as np
import pandas as pd
import json

# Load the dataset
with open("../../datasets/history-250K.json", "r") as f:
    data = json.load(f)

# Prepare the data
color_chars_map = {"white": "B", "black": "P", "red": "V"}
resultados = [color_chars_map[entry["color"]] for entry in data]
cores = {"V": 1, "P": 0, "B": 0}
df = pd.DataFrame({"resultado": resultados, "cor": [cores[r] for r in resultados]})

# Split the data into training and testing sets
train_df = df.iloc[:249000]
test_df = df.iloc[249000:]

# Create the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(8, activation="sigmoid", input_shape=(1,)),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model with verbose=1 to show progress
model.fit(
    train_df["cor"],
    tf.keras.utils.to_categorical(train_df.index % 3, num_classes=3),
    epochs=10,
    verbose=1,  # Show progress
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)],
)

# Evaluate the model
loss, accuracy = model.evaluate(
    test_df["cor"], tf.keras.utils.to_categorical(test_df.index % 3, num_classes=3)
)
print(f"Test accuracy: {accuracy:.2f}%")
