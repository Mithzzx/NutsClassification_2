import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
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
        img = img.flatten()  # Flatten the image
        images.append(img)
    return np.array(images)

# Encode the labels
label_encoder = LabelEncoder()
data['labels'] = label_encoder.fit_transform(data['labels'])

# Split the data into training, validation, and test sets
train_data = data[data['data set'] == 'train']
valid_data = data[data['data set'] == 'valid']
test_data = data[data['data set'] == 'test']

# Load images and labels
X_train = load_images(train_data['filepaths'])
y_train = train_data['labels']
X_valid = load_images(valid_data['filepaths'])
y_valid = valid_data['labels']
X_test = load_images(test_data['filepaths'])
y_test = test_data['labels']

# Train the SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Validate the model
y_valid_pred = svm_model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print(f'Validation Accuracy: {valid_accuracy * 100:.2f}%')

# Test the model
y_test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save the model
joblib.dump(svm_model, 'nuts_classification_svm_model.pkl')