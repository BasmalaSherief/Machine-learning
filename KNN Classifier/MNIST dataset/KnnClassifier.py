#!/venv/bin/python3
from scipy import stats
import numpy as np

class Knn:
    
    def __init__ (self, k):
        print("Prediction...")
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        X_test = np.array(X_test)
        all_predictions = []
        
        # Loop through each test point
        for i, test_point in enumerate(X_test):
            
            # calculate distance to ALL training points at once.
            distances = np.linalg.norm(self.X_train - test_point, axis=1)
            
            # np.argsort returns the INDICES of the sorted elements
            # We take the first k indices
            nearest_indices = np.argsort(distances)[:self.k]
            
            # Get the labels for those indices
            neighbor_labels = self.y_train[nearest_indices]
            
            # Find the most common label
            prediction = stats.mode(neighbor_labels).mode
            
            all_predictions.append(prediction)
            
            # Print progress every 500 images to check if it's not frozen
            if i % 500 == 0:
                print(f"Processing image {i}/{len(X_test)}...")
            
        return np.array(all_predictions)

    def test(self, y_test, y_pred):
        return np.mean(y_test == y_pred)
    
    def __del__(self):
        print("Prediction ended") 