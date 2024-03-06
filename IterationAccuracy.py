import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulated data with an accuracy of 80%
np.random.seed(42)
num_samples = 1000
num_iterations = 5

# Simulated features and target variable
X = np.random.rand(num_samples, 5)  # Replace with your features
y = np.random.randint(0, 2, num_samples)  # Replace with your target

# Model evolution over iterations
accuracy_results = []

for i in range(num_iterations):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Store model performance
    accuracy_results.append(accuracy)
    print(f"Iteration {i + 1} - Accuracy: {accuracy:.4f}")

    # If accuracy reaches 80%, break the loop
    if accuracy >= 0.80:
        print(f"Desired accuracy of 80% achieved in iteration {i + 1}.")
        break

# Plotting model evolution
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(accuracy_results) + 1), accuracy_results, marker='o', linestyle='-')
plt.title('Model Evolution Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()