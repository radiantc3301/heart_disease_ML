import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.1, max_iterations=1000, epsilon = 1e-8):
    """
    Implements logistic regression for binary classification.
    
    Args:
        X (np.ndarray): Input features (2D array).
        y (np.ndarray): Target labels (1D array).
        learning_rate (float): Learning rate for weight updates.
        max_iterations (int): Maximum number of iterations.
        epsilon (float): Small value to prevent division by zero.
        
    Returns:
        np.ndarray: Learned weight vector.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    
    for _ in range(max_iterations):
        z = np.dot(X, weights)
        y_pred = sigmoid(z)
        gradient = np.dot(X.T, y_pred - y) / n_samples
        weights -= learning_rate * gradient
        
        if np.linalg.norm(gradient) < epsilon:
            break

    
    return weights

# Load the data into a pandas DataFrame
data = pd.read_csv('heart.csv')

# Separate the input features (X) and target labels (y)
X = data.drop('target', axis=1).values
y = data['target'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the perceptron model
weights = logistic_regression(X_train, y_train)

# Evaluate the model on the test set
correct = 0
incorrect = 0

for i in range(len(X_test)):
    if sigmoid(np.dot(X_test[i], weights)) < 0.5:
        y_pred = 0
    else:
        y_pred = 1
    if y_pred != y_test[i]:
        incorrect += 1
    else:
        correct += 1
    
print("Accuracy: ", correct / (correct + incorrect) * 100, "%")