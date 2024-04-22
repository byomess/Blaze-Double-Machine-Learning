import tensorflow as tf
import numpy as np
import pandas as pd
import json

# Load the dataset
with open("../../datasets/history-250K.json", "r") as f:
    json_history = json.load(f)

# Prepare the data
results = []
for i in range(len(json_history)):
    color_chars_map = {"white": "B", "black": "P", "red": "V"}
    results.append(color_chars_map[json_history[i]["color"]])

if len(results) < 249000:
    print("Insufficient data for training.")
    exit()

# Define the color encoding
color_encoding = {"V": 1, "P": 0, "B": 0}

# Create the dataframe
df = pd.DataFrame({"result": results, "color": [color_encoding[r] for r in results]})

# Split the data into training and test sets
train_df = df.iloc[:249000]
test_df = df.iloc[249000:]

# Define the model architecture
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
    train_df["color"],
    tf.keras.utils.to_categorical(train_df.index % 3, num_classes=3),
    epochs=1000,
    verbose=0,
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(
    test_df["color"],
    tf.keras.utils.to_categorical(test_df.index % 3, num_classes=3),
    verbose=0,
)

print(f"Test accuracy: {test_accuracy:.4f}")

# Use the trained model to predict the next color of the roulette
last_result = test_df.iloc[-1]["color"]
proximo_resultado = np.argmax(model.predict(np.array([last_result])), axis=-1)

# Print the prediction with the probability
probs = model.predict(np.array([last_result]))
if proximo_resultado == 0:
    print(f"Próximo resultado: preto, probability: {probs[0][0]:.4f}")
elif proximo_resultado == 1:
    print(f"Próximo resultado: vermelho, probability: {probs[0][1]:.4f}")
else:
    print(f"Próximo resultado: branco, probability: {probs[0][2]:.4f}")
