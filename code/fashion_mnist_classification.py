import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # If you need to further split
from sklearn.metrics import classification_report, confusion_matrix
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
num_samples = 10000  # Number of training samples to use
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