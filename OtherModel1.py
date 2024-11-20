import cv2
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from PIL import Image

# Data loading and preprocessing (similar to your original setup)
image_directory = 'datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')
dataset = []
label = []

INPUT_SIZE = 64

for image_name in no_tumor_images:
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for image_name in yes_tumor_images:
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

# Split and normalize data
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Load VGG16 model for feature extraction
feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
x_train_features = feature_extractor.predict(x_train)
x_test_features = feature_extractor.predict(x_test)

# Flatten the features for SVM
x_train_flat = x_train_features.reshape(x_train_features.shape[0], -1)
x_test_flat = x_test_features.reshape(x_test_features.shape[0], -1)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(x_train_flat, y_train)

# Predict and evaluate
y_pred = svm_model.predict(x_test_flat)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")
