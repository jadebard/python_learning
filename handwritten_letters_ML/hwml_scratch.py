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
m,n = data.shape # m is number of examples, n is number of features (including label)
np.random.shuffle(data) # shuffle the data to make the training more representative of the overall set.

# Alphabet mapping, essentially maps the label 0-25 to the corresponding letter A-Z.
alphabet = [chr(ord('A') + i) for i in range(26)]

# Separating our data. _dev data is what we will use to evaluate our model during development, and _train data is what we will use to train our model. 
# We are using the first 1000 examples for development and the rest for training. 
# We also transpose the data so that each column represents an example and each row represents a feature (or label).
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_,m_train = X_train.shape


def forward_prop(W1, b1, W2, b2, X):
   Z1 = W1.dot(X) + b1    # unactivated hidden layer (1st layer) takes the weight defined in init_params and dot's to the input matrix X then adds the bias (b1)
   A1 = ReLU(Z1)          # rectified linear unit activation function applied to the hidden layer
   Z2 = W2.dot(A1) + b2
   A2 = softmax(Z2)
   return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    dZ2 = A2 - one_hot(Y) # error at output layer, one_hot converts the labels into a one-hot encoded format (e.g., label 0 becomes [1, 0, 0, ..., 0], label 1 becomes [0, 1, 0, ..., 0], etc.)
    dW2 = dZ2.dot(A1.T) / m # gradient for W2, we take the dot product of the error at the output layer and the activations from the hidden layer, then average over the number of training examples
    dB2 = np.sum(dZ2) / m # gradient for b2, we sum the error at the output layer across all examples and average)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) # error at hidden layer, we backpropagate the error from the output layer to the hidden layer by taking the dot product of the transposed weights of the output layer and the error at the output layer, then we multiply element-wise by the derivative of the ReLU activation function applied to Z1 (the unactivated hidden layer)
    dW1 = dZ1.dot(X.T) / m # gradient for W1, we take the dot product of the error at the hidden layer and the activations from the input layer (which is just X), then average
    dB1 = np.sum(dZ1) / m # gradient for b1, we sum the error at the hidden layer across all examples and average
    return dW1, dB1, dW2, dB2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # create a matrix of zeros with shape (number of examples, number of classes). Y.size is the number of examples, Y.max() + 1 is the number of classes (since labels start at 0)
    one_hot_Y[np.arange(Y.size), Y] = 1 # set the appropriate elements to 1 based on the labels in Y. np.arange(Y.size) creates an array of indices from 0 to number of examples - 1, and Y contains the class labels for each example. This line effectively creates a one-hot encoding of the labels.
    one_hot_Y = one_hot_Y.T # transpose the one-hot matrix so that each column represents an example and each row represents a class
    return one_hot_Y

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True)) # subtract max for numerical stability
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# Rectified Linear Unit activation function, it outputs the input directly if it is positive, otherwise, it will output zero. 
# This is a common activation function used in hidden layers of neural networks.
def ReLU(Z):
    return np.maximum(0,Z)

def ReLU_deriv(Z):
    return Z > 0

def init_params():
    W1 = np.random.rand(10, 784) - 0.5 # 10 hidden units, 784 input features (28x28 pixels)
    b1 = np.random.rand(10, 1) - 0.5 # bias for hidden layer
    W2 = np.random.rand(26, 10) - 0.5 # 26 output classes (A-Z), 10 hidden units
    b2 = np.random.rand(26, 1) - 0.5 # bias for output layer
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0) # returns the index of the maximum value in each column of A2, which corresponds to the predicted class label for each example


def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)


def update_params(W1, b1, W2, b2, dW1, dB1, dW2, dB2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * dB2
    return W1, b1, W2, b2

def gradient_descent(X, Y, learning_rate, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, dB1, dW2, dB2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, dB1, dW2, dB2, learning_rate)
        if i % 100 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f'Iteration {i}, Accuracy: {accuracy:.4f}')
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.mean(predictions == Y)

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)