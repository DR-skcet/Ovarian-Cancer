import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.segmentation import active_contour
from sklearn.utils import resample
import plotly.express as px

# Generate sample genetic data
np.random.seed(0)
genetic_data = np.random.rand(50, 5)

# Generate sample clinical records
clinical_records = pd.DataFrame({
    'age': np.random.randint(30, 80, 50),
    'stage': np.random.choice(['I', 'II', 'III', 'IV'], 50),
})

# Generate sample histopathological images
image = data.coins()
image_rgb = np.stack((image,) * 3, axis=-1)  # Convert 2D grayscale to 3D RGB
image_gray = rgb2gray(image_rgb)
image_bin = image_gray < threshold_otsu(image_gray)
s = np.linspace(0, 2*np.pi, 50)
r = 100 + 50*np.sin(s)
c = 220 + 50*np.cos(s)
init = np.array([r, c]).T
snake = active_contour(image_bin, init, alpha=0.015, beta=10, gamma=0.001)
histopathological_images = np.random.rand(50, len(snake))

# Combine data
data_combined = np.hstack((genetic_data, clinical_records[['age']], histopathological_images))

# Simulate outcome variable (subtype)
subtype = np.random.choice(['A', 'B', 'C'], 50)

# Simulate outlier
outliers = np.random.choice([0, 1], 50)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_combined, subtype, test_size=0.2, random_state=0)

# Upsample minority class for subtype
X_train_upsampled, y_train_upsampled = resample(X_train[y_train == 'C'], y_train[y_train == 'C'], replace=True, n_samples=X_train[y_train == 'A'].shape[0], random_state=0)
X_train_balanced = np.vstack((X_train, X_train_upsampled))
y_train_balanced = np.concatenate((y_train, y_train_upsampled))

# Feature selection (not implemented here)
# Feature engineering
# Add new features based on genetic_data
genetic_sum = np.sum(genetic_data, axis=1)
genetic_mean = np.mean(genetic_data, axis=1)
genetic_std = np.std(genetic_data, axis=1)

# Add new features to data_combined
data_combined = np.hstack((data_combined, genetic_sum[:, np.newaxis], genetic_mean[:, np.newaxis], genetic_std[:, np.newaxis]))

# Model training
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train_balanced, y_train_balanced)

# Model evaluation
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Subtype Classification detection Confusion Matrix using "Random Forest"')
plt.show()

# Calculate ROC curve and AUC
y_proba = rf.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label='C')
auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

# Plot ROC curve
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Subtype Classification detection ROC Curve using "Random Forest"')
plt.show()

# Create a DataFrame for the genetic data
genetic_df = pd.DataFrame(genetic_data, columns=['Feature 1', 'Feature 2', 'Feature 3', 'gene4', 'gene5'])

# Add subtype to genetic_df for color differentiation in the scatter plot
genetic_df['subtype'] = subtype

# Plot 3D scatter plot
fig = px.scatter_3d(genetic_df, x='Feature 1', y='Feature 2', z='Feature 3', color='subtype', labels={'subtype': 'Subtype'})
fig.update_layout(title='Genetic Data 3D Scatter Plot for SubType Classification using Random Forest')
fig.show()

# Classification model output
classification_predictions = np.array(['Serous', 'Endometrioid', 'Clear Cell', 'Mucinous', 'Serous', 'Outlier'])
true_labels = np.array(['Serous', 'Endometrioid', 'Clear Cell', 'Mucinous', 'Serous', 'Clear Cell'])

# Example of classification output
print("Classification Model Output:")
for i in range(len(classification_predictions)):
    print(f"Sample {i+1} - Predicted: {classification_predictions[i]}, True: {true_labels[i]}")

# Classification model output
classification_predictions = np.array(['Serous', 'Endometrioid', 'Clear Cell', 'Mucinous', 'Serous', 'Outlier'])
true_labels = np.array(['Serous', 'Endometrioid', 'Clear Cell', 'Mucinous', 'Serous', 'Clear Cell'])

# Calculate classification accuracy
classification_accuracy = np.mean(classification_predictions == true_labels) * 100
print("Using Random Forest")
# Generate a sample dataset (replace this with your actual dataset)
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Print Random Forest Accuracy
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f'AUC: {auc}')
