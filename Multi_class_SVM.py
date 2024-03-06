import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix
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
genetic_data = np.random.rand(200, 5)

# Generate sample clinical records
clinical_records = pd.DataFrame({
    'age': np.random.randint(30, 80, 200),
    'stage': np.random.randint(0, 5, 200),  # Simulate stages ranging from 0 to 4
})

# Generate sample histopathological images
image = data.coins()
image_rgb = np.stack((image,) * 3, axis=-1)  # Convert 2D grayscale to 3D RGB
image_gray = rgb2gray(image_rgb)
image_bin = image_gray < threshold_otsu(image_gray)
s = np.linspace(0, 2*np.pi, 200)
r = 100 + 50*np.sin(s)
c = 220 + 50*np.cos(s)
init = np.array([r, c]).T
snake = active_contour(image_bin, init, alpha=0.015, beta=10, gamma=0.001)
histopathological_images = np.random.rand(200, len(snake))

# Combine data
data_combined = np.hstack((genetic_data, clinical_records[['age', 'stage']], histopathological_images))

# Split data into training and testing sets for stage classification
X_train, X_test, y_train, y_test = train_test_split(data_combined, clinical_records['stage'], test_size=0.2, random_state=0)

# Upsample minority classes for stage classification
X_train_balanced = np.concatenate([X_train] + [X_train[y_train == i] for i in range(5)])
y_train_balanced = np.concatenate([y_train] + [y_train[y_train == i] for i in range(5)])

# Model training for stage classification
svm = SVC(kernel='linear', decision_function_shape='ovo', random_state=0)
svm.fit(X_train_balanced, y_train_balanced)

# Model evaluation for stage classification
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix for stage classification
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Stage Classification Confusion Matrix')
plt.show()

print(f'Stage Classification Accuracy: {accuracy}')

# Model evaluation for stage classification
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate probabilities for each class
y_probs = svm.decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_probs[:, i])
    roc_auc[i] = roc_auc_score((y_test == i).astype(int), y_probs[:, i])

# Plot ROC curve
plt.figure()
plt.plot(fpr[2], tpr[2], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Stage 2')
plt.legend(loc="lower right")
plt.show()

# print(f'Stage Classification Accuracy: {accuracy}')


# Generate some example data
np.random.seed(0)
df = pd.DataFrame({
    'Feature0': np.random.rand(100),
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'stage': np.random.randint(0, 5, 100)  # Stages ranging from 0 to 4
})

# Map the stage values to 'Stage 0', 'Stage 1', ..., 'Stage 4'
stage_map = {0: 'Stage 0', 1: 'Stage 1', 2: 'Stage 2', 3: 'Stage 3', 4: 'Stage 4'}
df['stage_label'] = df['stage'].map(stage_map)

# Specify the category order for stage_label in ascending order
category_order = ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
df['stage_label'] = pd.Categorical(df['stage_label'], categories=category_order, ordered=True)

# Create a 3D interactive scatter plot
fig = px.scatter_3d(df, x='Feature0', y='Feature1', z='Feature2', color='stage_label',
                     symbol='stage_label', opacity=0.7,
                     labels={'Feature0': 'Feature 0', 'Feature1': 'Feature 1', 'Feature2': 'Feature 2', 'stage_label': 'Stage'},
                     title='Genetic Data 3D Scatter Plot for Cancer Stages')
fig.update_traces(marker=dict(size=5),
                  selector=dict(mode='markers'))
fig.show()


