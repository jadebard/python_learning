import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# 1) Load dataset
csv_path = '.\handwritten_letters_ML\A_Z Handwritten Data.csv'
data = pd.read_csv(csv_path)
data = np.array(data)
np.random.shuffle(data) # shuffle the data to make the training more representative of the overall set.

# Alphabet mapping
alphabet = [chr(ord('A') + i) for i in range(26)]

# 2) Dataset structure: col 0 label 0-25, rest are 784 pixel features
labels = data[:, 0].astype(int)
X = data[:, 1:].astype(float)
y = labels

print('X shape', X.shape, 'labels shape', labels.shape)
print('label counts:', np.bincount(labels))

# 3) Visualize some example images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    label = labels[i]
    pixels = X[i].reshape(28, 28)
    axes[i//5, i%5].imshow(pixels, cmap='gray')
    axes[i//5, i%5].set_title(f'Label: {alphabet[label]}')
    axes[i//5, i%5].axis('off')
plt.tight_layout()
plt.show()

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print('train', X_train.shape, 'test', X_test.shape)

# 4) Scale (optional but good for many models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5) Fit baseline model (random forest multiclass)
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# 6) Evaluate
y_pred = clf.predict(X_test)
print('\nAccuracy:', accuracy_score(y_test, y_pred))
print('\nClassification report:\n', classification_report(y_test, y_pred, digits=4))
print('\nConfusion matrix (shape', confusion_matrix(y_test, y_pred).shape, '):')
print(confusion_matrix(y_test, y_pred))

# 7) Save model and scaler for later inference
joblib.dump(clf, 'letter_clf.pkl')
joblib.dump(scaler, 'scaler.pkl')
print('\nSaved model to letter_clf.pkl and scaler to scaler.pkl')

# 8) Visualize predictions on test samples
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    idx = np.random.randint(0, len(X_test)) 
    sample_pixels = X_test[idx]
    true_label = y_test[idx]
    pred_label = clf.predict(sample_pixels.reshape(1, -1))[0]
    pixels_reshaped = sample_pixels.reshape(28, 28)
    axes[i//5, i%5].imshow(pixels_reshaped, cmap='gray')
    axes[i//5, i%5].set_title(f'True: {alphabet[true_label]}\nPred: {alphabet[pred_label]}')
    axes[i//5, i%5].axis('off')
plt.tight_layout()
plt.show()

# 9) Helper for predict letter from flat pixel values
# alphabet = [chr(ord('A') + i) for i in range(26)]  # moved up

def predict_letter(sample_vector):
    sample_vector = np.array(sample_vector).reshape(1, -1)
    sample_vector = scaler.transform(sample_vector)
    pred = clf.predict(sample_vector)[0]
    return alphabet[pred]

# quick self-check with first test item
sample_idx = np.random.randint(0, len(X_test))
print('\nExample test prediction', predict_letter(X_test[sample_idx]), 'actual', alphabet[y_test[sample_idx]])
