import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Sample dataset (text inputs and their corresponding colors)
data = [
    ("red", "red"),
    ("blue", "blue"),
    ("yellow", "yellow"),
    ("green", "green"),
    ("orange", "orange"),
    ("purple", "purple"),
    ("white", "white"),
    ("black", "black")
]

# Create a mapping from color names to numerical labels
color_to_label = {color: idx for idx, (text, color) in enumerate(data)}

# Create the training data (text inputs and their corresponding numerical labels)
texts, labels = zip(*data)
encoded_labels = [color_to_label[label] for label in labels]

# Encode the labels using LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(encoded_labels)

# Convert the text inputs to lowercase for consistency
texts = [text.lower() for text in texts]

# Create a mapping from characters to numerical values (one-hot encoding)
char_to_idx = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz")}

def text_to_one_hot(text, max_length=10):
    # Convert text to lowercase and pad to max_length
    text = text.lower().ljust(max_length)
    # Convert characters to their numerical values using one-hot encoding
    return [char_to_idx[char] for char in text]

# Convert text inputs to one-hot encoded arrays
encoded_texts = np.array([text_to_one_hot(text) for text in texts])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_texts, encoded_labels, test_size=0.2, random_state=42)

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='softmax')  # 8 output neurons for 8 colors
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=4)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
