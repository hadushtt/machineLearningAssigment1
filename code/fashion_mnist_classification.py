import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # If you need to further split [cite: 10]
from sklearn.metrics import classification_report, confusion_matrix #For classification [cite: 10]
import tensorflow as tf  # Or torch, if you prefer PyTorch
from tensorflow.keras.datasets import fashion_mnist  # If using TensorFlow/Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D  # Example layers

# Define class names (for better visualization)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the Fashion-MNIST dataset from Keras
(train_images_full, train_labels_full), (test_images, test_labels) = fashion_mnist.load_data()

# --- Sampling the data (as per the assignment recommendation) ---
num_samples = 10000  # Number of training samples to use.  Group 9 is assigned Fashion-MNIST Dataset (Sample ~10K images) [cite: 8]
train_images = train_images_full[:num_samples]
train_labels = train_labels_full[:num_samples]

# --- Data Inspection (as part of EDA) ---
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

# --- Visualization (first part of EDA) ---
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])  # Use class_names for labels
plt.show()

# Show class distribution (example)
unique_labels, counts = np.unique(train_labels, return_counts=True)
plt.bar(unique_labels, counts)
plt.xlabel('Class (0-9)')
plt.ylabel('Number of Samples')
plt.title('Distribution of Classes in Training Set')
plt.xticks(range(10))
plt.show()

# --- Preprocessing ---
# Normalize pixel values.  This is a recommended preprocessing step [cite: 10]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape for CNN (add channel dimension). CNN is chosen as an appropriate model for image data.
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# --- Model Training ---
# Model Selection: CNN (Convolutional Neural Network)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes
])

# Model Compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if labels are one-hot encoded
              metrics=['accuracy'])

# Model Training
#Training details (hyperparameters, validation strategy). [cite: 13]
model.fit(train_images, train_labels, epochs=10, validation_split=0.2)  # Example: 10 epochs.  Validation strategy is train_test_split [cite: 10]

# --- Prediction & Evaluation ---
# Make predictions
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability

# Evaluation:  Evaluation using Accuracy, Precision, Recall, F1-score, Confusion Matrix. [cite: 10]
print("\nEvaluation Metrics:")
print(classification_report(test_labels, y_pred_classes, target_names=class_names))  # Class-wise metrics
print("\nConfusion Matrix:")
print(confusion_matrix(test_labels, y_pred_classes))

# --- Conclusion (in code - for quick view) ---
print("\nModel trained and evaluated on Fashion-MNIST dataset.") #Summarize findings [cite: 15]