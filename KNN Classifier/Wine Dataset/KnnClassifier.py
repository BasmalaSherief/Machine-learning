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
        # A list to store the final prediction for each test point
        all_predictions = []
        
        for test_point in X_test:
            # A list to store distances for this test point
            distances = []
            
            for i in range(len(self.X_train)):
                train_point = self.X_train[i]
                train_label = self.y_train[i]
                
                # Calculate the Euclidean distance
                distance = np.linalg.norm(test_point - train_point)
                
                distances.append((distance, train_label))
            
            # Sort the list of tuples by distance (the first item in the tuple)
            sorted_distances = sorted(distances, key=lambda x: x[0])
            
            # Get just the top 'k' items from the sorted list
            k_nearest = sorted_distances[:self.k]
            
            # Extract just the labels 
            neighbor_labels = [item[1] for item in k_nearest]
            
            # Find the most common label among the neighbors
            prediction = stats.mode(neighbor_labels).mode
            
            # Add this one prediction to our main list
            all_predictions.append(prediction)
            
        # Return the full list of predictions as a NumPy array
        return np.array(all_predictions)

    def test(self, y_test, y_pred):
        return np.mean(y_test == y_pred)
    
    def __del__(self):
        print("Prediction ended") 