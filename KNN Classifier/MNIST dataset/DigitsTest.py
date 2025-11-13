#!/venv/bin/python3
# the classifier
from KnnClassifier import Knn 
import numpy as np
#  data loader
import tensorflow
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

#load the data
(X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

# Reshape the data 
# Divided by 255.0 to get values between 0 and 1.
X_train_full = X_train_full.reshape(60000, 784) / 255.0
X_test_full = X_test_full.reshape(10000, 784) / 255.0

# Define the list of k's to test ---
k_values = [1, 3, 5, 10] 

# A dictionary to store final results
all_results = {}

# Iterate over each k
for k in k_values:
    print(f"--- Testing for k = {k} ---")
    
    accuracies_for_this_k = []

    # ---  The 30 Experiments Loop (6 x 5) ---
    
    # Create 6  training sets
    for i in range(6): 
        
        # Create a 10,000-image training subset
        # Get 10,000 random, unique indices from the 60,000
        train_indices = np.random.choice(len(X_train_full), 10000, replace=False)
        
        # Slice the full data to get subset
        X_train_subset = X_train_full[train_indices]
        y_train_subset = y_train_full[train_indices]
        
        # Create 5 different test sets
        for j in range(5): 
            
            # Create a 2,000-image test subset
            # Get 2,000 random, unique indices from the 10,000
            test_indices = np.random.choice(len(X_test_full), 2000, replace=False)
            
            X_test_subset = X_test_full[test_indices]
            y_test_subset = y_test_full[test_indices]
            
            model = Knn(k=k)
            model.fit(X_train_subset, y_train_subset)
            y_pred = model.predict(X_test_subset)
            
            # accuracy
            acc = model.test(y_test_subset, y_pred)
            accuracies_for_this_k.append(acc)


    # --- Statistics for this k ---
    
    # Mean
    mean_accuracy = np.mean(accuracies_for_this_k) 
    
    # Standard deviation
    std_deviation = np.std(accuracies_for_this_k)
    
    print(f"k={k}: Mean Accuracy = {mean_accuracy:.4f}, Spread (Std Dev) = {std_deviation:.4f}")
    all_results[k] = {'mean': mean_accuracy, 'std': std_deviation}

# plotting 
# Get the sorted list of k values 
k_list = sorted(all_results.keys())

# Extract the means and standard deviations in the same order
mean_accuracies = [all_results[k]['mean'] for k in k_list]
std_deviations = [all_results[k]['std'] for k in k_list]

plt.figure(figsize=(10, 6))
plt.errorbar(k_list, mean_accuracies, yerr=std_deviations, 
             fmt='-o', capsize=5, ecolor='red', color='blue', label='Mean Accuracy')

plt.title('KNN Performance on MNIST: Accuracy vs. k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()