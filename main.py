import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def perceptron_learning(X, y, learning_rate=0.1, max_iterations=1000):
    """
    Implements the perceptron learning algorithm for binary classification.
    
    Args:
        X (np.ndarray): Input features (2D array).
        y (np.ndarray): Target labels (1D array).
        learning_rate (float): Learning rate for weight updates.
        max_iterations (int): Maximum number of iterations for learning.
        
    Returns:
        np.ndarray: Learned weight vector.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    
    for _ in range(max_iterations):
        misclassified = False
        for i in range(n_samples):
            if(np.dot(X[i], weights) < 0):
                prediction = 0
            else:
                prediction = 1
            if prediction != y[i]:
                misclassified = True
                weights += learning_rate * (y[i] - prediction) * X[i]
        if not misclassified:
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
weights = perceptron_learning(X_train, y_train)

# Evaluate the model on the test set
correct = 0
incorrect = 0

for i in range(len(X_test)):
    if np.dot(X_test[i], weights) < 0:
        y_pred = 0
    else:
        y_pred = 1
    if y_pred != y_test[i]:
        incorrect += 1
    else:
        correct += 1
    
print("Accuracy: ", correct / (correct + incorrect) * 100, "%")