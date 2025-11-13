#import the classifier I implemented
from KnnClassifier import Knn 
import numpy as np
# import data loader
from sklearn.datasets import load_wine
#import normalization 
from sklearn.preprocessing import MinMaxScaler
# import performance metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import utility function to split the dataset
from sklearn.model_selection import train_test_split
# import plotting
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# Load and prep the Wine data 
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)


k_value = 15
model = Knn(k=k_value)

# Fit the model
model.fit(X_train_norm, y_train)

# Make predictions
y_pred = model.predict(X_test_norm)

# Test the model's accuracy
accuracy = model.test(y_test, y_pred)

print(f"--- My KNN Model (k={k_value}) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

# Plotting
pca = PCA(n_components=2)
# Fit on the training data
pca.fit(X_train_norm) 
X_pca = pca.transform(X_test_norm) 
X_error = X_pca[y_test != y_pred, :]
colors = ['red','green','blue'] 

plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:,0], X_pca[:,1], s=12, marker='x', c=y_test, cmap=ListedColormap(colors))
plt.plot(X_error[:,0], X_error[:,1], 'ok', markersize=15, fillstyle='none', label='Misclassified')
plt.title(f"KNN (k={k_value}) Predictions (Test Set)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()
