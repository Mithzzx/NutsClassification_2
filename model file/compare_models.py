import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Read the CSV file
csv_file = 'path/to/tree%20nuts.csv'
data = pd.read_csv(csv_file)

# Load and preprocess the images
def load_images(filepaths):
    images = []
    for filepath in filepaths:
        img = cv2.imread(filepath)
        img = cv2.resize(img, (150, 150))
        img = img.flatten()  # Flatten the image for SVM
        images.append(img)
    return np.array(images)

# Encode the labels
label_encoder = LabelEncoder()
data['labels'] = label_encoder.fit_transform(data['labels'])

# Split the data into test set
test_data = data[data['data set'] == 'test']

# Load images and labels
X_test = load_images(test_data['filepaths'])
y_test = test_data['labels']

# Load the SVM model
svm_model = joblib.load('nuts_classification_svm_model.pkl')

# Evaluate the SVM model
y_test_pred_svm = svm_model.predict(X_test)
svm_test_accuracy = accuracy_score(y_test, y_test_pred_svm)
print(f'SVM Test Accuracy: {svm_test_accuracy * 100:.2f}%')

# Preprocess images for CNN model
def preprocess_images_for_cnn(filepaths):
    images = []
    for filepath in filepaths:
        img = cv2.imread(filepath)
        img = cv2.resize(img, (150, 150))
        img = img / 255.0  # Normalize the image
        images.append(img)
    return np.array(images)

X_test_cnn = preprocess_images_for_cnn(test_data['filepaths'])

# Load the CNN model
cnn_model = tf.keras.models.load_model('nuts_classification_model.h5')

# Evaluate the CNN model
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
print(f'CNN Test Accuracy: {cnn_test_accuracy * 100:.2f}%')