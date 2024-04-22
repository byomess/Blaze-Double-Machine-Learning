import tensorflow as tf
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Load the dataset
with open("../../datasets/history-250K.json", "r") as f:
    data = json.load(f)

# Prepare the data
color_chars_map = {"white": 0, "black": 1, "red": 2}
resultados = [color_chars_map[entry["color"]] for entry in data]

df = pd.DataFrame({
    "resultado": resultados,
    "cor": [color_chars_map[entry["color"]] for entry in data],
    "number": [entry["number"] for entry in data],
    "players": [entry["players"] for entry in data],
    "redbets": [entry["redbets"] for entry in data],
    "whitebets": [entry["whitebets"] for entry in data],
    "blackbets": [entry["blackbets"] for entry in data]
})

# Scale the data
scaler = StandardScaler()
df[["number", "players", "redbets", "whitebets", "blackbets"]] = scaler.fit_transform(df[["number", "players", "redbets", "whitebets", "blackbets"]])

# Split the data into features and target variable
X = df.drop("resultado", axis=1).values
y = df["resultado"].values

# Use the first 100k entries for training and the last 100 entries for testing
X_train = X[:1000]
y_train = y[:1000]
X_test = X[-10:]
y_test = y[-10:]

# Define the model
def create_model(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="sigmoid", input_shape=(6,)),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

# Wrap the model with KerasClassifier
model = KerasClassifier(build_fn=create_model)

# Define hyperparameter space
param_grid = {
    "optimizer": ["adam", "rmsprop", "sgd"],
    "epochs": [5, 10, 20],
    "batch_size": [2, 4, 8]
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_result = grid.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_result.best_estimator_
best_keras_model = best_model.model

# Evaluate the best model
loss, accuracy = best_keras_model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}%")

# Save the best model to disk
best_keras_model.save("best_model.h5")
print("Model saved to disk.")

# Delete the model object from memory
del best_model
