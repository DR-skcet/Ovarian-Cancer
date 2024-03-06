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
import seaborn as sns

# Generate sample genetic data
np.random.seed(0)
genetic_data = np.random.rand(100, 5)

# Generate sample clinical records
clinical_records = pd.DataFrame({
    'age': np.random.randint(30, 80, 100),
    'stage': np.random.choice(['I', 'II', 'III', 'IV'], 100),
})

# Generate sample histopathological images
image = data.coins()
image_rgb = np.stack((image,) * 3, axis=-1)  # Convert 2D grayscale to 3D RGB
image_gray = rgb2gray(image_rgb)
image_bin = image_gray < threshold_otsu(image_gray)
s = np.linspace(0, 2*np.pi, 100)
r = 100 + 50*np.sin(s)
c = 220 + 50*np.cos(s)
init = np.array([r, c]).T
snake = active_contour(image_bin, init, alpha=0.015, beta=10, gamma=0.001)
histopathological_images = np.random.rand(100, len(snake))

# Combine data
data_combined = np.hstack((genetic_data, clinical_records[['age']], histopathological_images))

# Simulate outcome variable (subtype)
subtype = np.random.choice(['A', 'B', 'C'], 100)

# Simulate outlier
outliers = np.random.choice([0, 1], 100)

# Split data into training and testing sets for subtype classification
X_train, X_test, y_train, y_test = train_test_split(data_combined, subtype, test_size=0.2, random_state=0)

# Upsample minority class for subtype classification
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

# Model training for subtype classification
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train_balanced, y_train_balanced)

# Model evaluation for subtype classification
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Split data into training and testing sets for outlier detection
X_train_outlier, X_test_outlier, y_train_outlier, y_test_outlier = train_test_split(data_combined, outliers, test_size=0.2, random_state=0)

# Upsample minority class for outliers
X_train_outlier_upsampled, y_train_outlier_upsampled = resample(X_train_outlier[y_train_outlier == 1], y_train_outlier[y_train_outlier == 1], replace=True, n_samples=X_train_outlier[y_train_outlier == 0].shape[0], random_state=0)
X_train_outlier_balanced = np.vstack((X_train_outlier, X_train_outlier_upsampled))
y_train_outlier_balanced = np.concatenate((y_train_outlier, y_train_outlier_upsampled))

# Model training for outlier detection
svm = SVC(kernel='linear', random_state=0)
svm.fit(X_train_outlier_balanced, y_train_outlier_balanced)

# Model evaluation for outlier detection
y_pred_outlier = svm.predict(X_test_outlier)
accuracy_outlier = accuracy_score(y_test_outlier, y_pred_outlier)
conf_matrix_outlier = confusion_matrix(y_test_outlier, y_pred_outlier)

# Plot confusion matrix for outlier detection
sns.heatmap(conf_matrix_outlier, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Outlier Detection Confusion Matrix using SVM')
plt.show()

# Calculate ROC curve and AUC for outlier detection
y_proba_outlier = svm.decision_function(X_test_outlier)
fpr_outlier, tpr_outlier, _ = roc_curve(y_test_outlier, y_proba_outlier)
auc_outlier = roc_auc_score(y_test_outlier, y_proba_outlier)

# Plot ROC curve for outlier detection
plt.plot(fpr_outlier, tpr_outlier, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Outlier Detection ROC Curve using SVM')
plt.show()

# Create a DataFrame for the genetic data
genetic_df_outlier = pd.DataFrame(data_combined, columns=[f'Feature {i}' for i in range(1, 6)] + ['age'] + [f'pixel_{i}' for i in range(len(snake))] + ['genetic_sum', 'genetic_mean', 'genetic_std'])

# Add outliers to genetic_df_outlier for color differentiation in the scatter plot
genetic_df_outlier['outlier'] = outliers

# Plot 3D scatter plot for outlier detection
fig_outlier = px.scatter_3d(genetic_df_outlier, x='Feature 1', y='Feature 2', z='Feature 3', color='outlier', labels={'outlier': 'Outlier'})
fig_outlier.update_layout(title='Genetic Data 3D Scatter Plot for Outlier Detection using SVM')
fig_outlier.show()

# Outlier detection model output
outlier_predictions = np.array([1, 1, 1, 1, -1, 1])
true_outliers = np.array([1, 1, 1, 1, -1, -1])

# Example of outlier detection output
print("\nOutlier Detection Model Output:")
for i in range(len(outlier_predictions)):
    if outlier_predictions[i] == 1:
        prediction = "Normal"
    else:
        prediction = "Outlier"

    if true_outliers[i] == 1:
        true_label = "Normal"
    else:
        true_label = "Outlier"

    print(f"Sample {i+1} - Predicted: {prediction}, True: {true_label}")

# Generate a sample dataset (replace this with your actual dataset)
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Support Vector Machine (SVM) Classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)

# Print SVM accuracy
print(f"SVM Accuracy: {svm_accuracy}")

print(f'Outlier Detection AUC: {auc_outlier}')
