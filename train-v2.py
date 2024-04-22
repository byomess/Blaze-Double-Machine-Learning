import tensorflow as tf
import numpy as np
import pandas as pd
import json

# Load the dataset
with open("../../datasets/dataset_file.json", "r") as f:
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

# Train the model
model.fit(
    train_df["cor"],
    tf.keras.utils.to_categorical(train_df.index % 3, num_classes=3),
    epochs=1000,
    verbose=0,
)

# Evaluate the model
predictions = model.predict(test_df["cor"])
proximo_resultado = np.argmax(predictions, axis=-1)
probabilities = np.max(predictions, axis=-1)

# Print the predictions and their probabilities
for i, result in enumerate(proximo_resultado):
    if result == 0:
        print(f"Próximo resultado: preto, probabilidade: {probabilities[i]:.2f}")
    elif result == 1:
        print(f"Próximo resultado: vermelho, probabilidade: {probabilities[i]:.2f}")
    else:
        print(f"Próximo resultado: branco, probabilidade: {probabilities[i]:.2f}")
