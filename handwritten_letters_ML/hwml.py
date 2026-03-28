import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# 1) Load dataset
csv_path = 'A_Z Handwritten Data.csv'
data = pd.read_csv(csv_path)
print('Loaded', data.shape, 'shape')
print(data.head())

# 2) Dataset structure: col '0' label 0-25, rest are 784 pixel features "0.1" ... "0.784"
label_col = '0'
X = data.drop(columns=[label_col])
y = data[label_col].astype(int)

print('X shape', X.shape, 'y distribution:\n', y.value_counts().sort_index())

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print('train', X_train.shape, 'test', X_test.shape)

# 4) Scale (optional but good for many models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5) Fit baseline model (logistic regression multiclass)
clf = LogisticRegression(
    multi_class='multinomial',
    solver='saga',
    max_iter=300,
    n_jobs=-1,
    random_state=42
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

# 8) Helper for predict letter from flat pixel values
alphabet = [chr(ord('A') + i) for i in range(26)]

def predict_letter(sample_vector):
    sample_vector = np.array(sample_vector).reshape(1, -1)
    sample_vector = scaler.transform(sample_vector)
    pred = clf.predict(sample_vector)[0]
    return alphabet[pred]

# quick self-check with first test item
sample_idx = 0
print('\nExample test prediction', predict_letter(X_test[sample_idx]), 'actual', alphabet[y_test.iloc[sample_idx]])
