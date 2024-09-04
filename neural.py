import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
import tkinter as tk

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode the target
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a custom callback to update the training progress in the Tkinter window
class TrainingProgressCallback(Callback):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def on_epoch_end(self, epoch, logs=None):
        progress_message = f"Epoch {epoch + 1}/{self.params['epochs']}\n"
        progress_message += f" - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}\n"
        progress_message += f" - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}\n\n"
        self.text_widget.insert(tk.END, progress_message)
        self.text_widget.see(tk.END)  # Scroll to the end

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes for the Iris dataset
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Create a Tkinter window
root = tk.Tk()
root.title("Training Progress")

# Create a Text widget to display the progress
text_widget = tk.Text(root, height=20, width=60)
text_widget.pack()

# Define the callback with the Text widget
progress_callback = TrainingProgressCallback(text_widget)

# Train the model with the custom callback
def start_training():
    model.fit(X_train, y_train, epochs=50, batch_size=5, validation_split=0.2, callbacks=[progress_callback])
    loss, accuracy = model.evaluate(X_test, y_test)
    final_message = f'Test Accuracy: {accuracy:.4f}\n'
    text_widget.insert(tk.END, final_message)

# Add a button to start training
start_button = tk.Button(root, text="Start Training", command=start_training)
start_button.pack()

# Function to save the model
def save_model():
    # Save the entire model
    model.save('iris_model.h5')
    text_widget.insert(tk.END, "Model saved successfully!\n")

# Add a button to save the model
save_button = tk.Button(root, text="Save Model", command=save_model)
save_button.pack()

# Run the Tkinter main loop
root.mainloop()
