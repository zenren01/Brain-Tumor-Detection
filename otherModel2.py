import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras.utils import normalize
from sklearn.metrics import accuracy_score
import joblib

image_directory = 'datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')
dataset = []
label = []

INPUT_SIZE = 64

# Load and preprocess images
for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

# Flatten images for MLP
dataset = dataset.reshape(len(dataset), -1)
dataset = normalize(dataset, axis=1)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Build MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, activation='relu', solver='adam', random_state=0)
mlp_model.fit(x_train, y_train)

# Evaluate model
y_pred = mlp_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("MLP Model Accuracy:", accuracy)

# Save MLP model

joblib.dump(mlp_model, 'BrainTumor_MLP_Model.pkl')
