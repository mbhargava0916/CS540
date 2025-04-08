import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename):
    """Load data from a CSV file."""
    data = pd.read_csv(filename)
    x = data['year'].values
    y = data['days'].values
    return x, y

def normalize_data(x):
    """Normalize the data using min-max normalization."""
    m = np.min(x)
    M = np.max(x)
    x_normalized = (x - m) / (M - m)
    return x_normalized, m, M

def augment_features(x_normalized):
    """Augment the normalized features with a column of ones."""
    n = len(x_normalized)
    X_augmented = np.column_stack((x_normalized, np.ones(n)))
    return X_augmented

def closed_form_solution(X_augmented, y):
    """Compute the closed-form solution for linear regression."""
    weights = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ y
    return weights

def gradient_descent(X_augmented, y, learning_rate, iterations):
    """Perform gradient descent to optimize the weights and bias."""
    n = len(y)
    weights = np.zeros(2)  # Initialize weights and bias to [0, 0]
    loss_history = []
    print(weights)

    for t in range(iterations):
        # Compute predictions
        y_pred = X_augmented @ weights
        # Compute gradient
        gradient = (1 / n) * X_augmented.T @ (y_pred - y)
        # Update weights
        weights -= learning_rate * gradient
        # Compute and store loss
        loss = np.mean((y_pred - y) ** 2) / 2
        loss_history.append(loss)
        # Print weights every 10 iterations
        if (t+1) % 10 == 0 and t!=iterations-1:
            print(weights)

    return weights, loss_history

def predict(x, weights, m, M):
    """Predict the number of ice days for a given year."""
    x_normalized = (x - m) / (M - m)
    y_pred = weights[0] * x_normalized + weights[1]
    return y_pred

def main():
    # Parse command-line arguments
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])

    # Load data
    x, y = load_data(filename)

    # Question 1: Data Visualization
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.savefig('data_plot.jpg')

    # Question 2: Data Normalization
    x_normalized, m, M = normalize_data(x)
    X_augmented = augment_features(x_normalized)
    print("Q2:")
    print(X_augmented)

    # Question 3: Closed-Form Solution
    weights_closed_form = closed_form_solution(X_augmented, y)
    print("Q3:")
    print(weights_closed_form)

    # Question 4: Gradient Descent
    print("Q4a:")
    weights_gd, loss_history = gradient_descent(X_augmented, y, learning_rate, iterations)
    print(f"Q4b: {learning_rate}")  # Replace with your learning rate
    print(f"Q4c: {iterations}")  # Replace with your number of iterations
    print(f"Q4d: I started with a learning_rate of {learning_rate} and increased it to 0.1 to ensure convergence within {iterations} iterations.")

    # Plot loss over time for Q4e
    plt.figure()
    plt.plot(range(iterations), loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.jpg')

    # Question 5: Prediction for 2023-24
    y_pred_2023 = predict(2023, weights_closed_form, m, M)
    print("Q5: " + str(y_pred_2023))

    # Question 6: Model Interpretation
    w = weights_closed_form[0]
    if w > 0:
        symbol = ">"
    elif w < 0:
        symbol = "<"
    else:
        symbol = "="
    print("Q6a: " + symbol)
    print("Q6b: If w > 0, ice days are increasing. If w < 0, ice days are decreasing. If w = 0, ice days remain constant.")

    # Question 7: Model Limitations
    x_star = m + (-weights_closed_form[1] * (M - m)) / weights_closed_form[0]
    print("Q7a: " + str(x_star))
    print("Q7b: This prediction assumes a linear trend, but climate change introduces non-linear effects making it unreliable.")

if __name__ == "__main__":
    main()