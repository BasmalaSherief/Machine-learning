from pandas import read_csv
import numpy as np
import tensorflow
import tensorflow.keras as tf_keras
import matplotlib.pyplot as plt

# read the data
data = read_csv('/home/basmala/Windows-Educational/Robotics Master/ML/Assignments/NeuralNetworkRegression/condition+based+maintenance+of+naval+propulsion+plants/UCI CBM Dataset/data.txt', sep='\s+').values

# split inputs (16) and targets (2)
X = data[:, 0:16]
Y = data[:, -2:]

# Data Normalizaion
#Observations
X_min = X.min(axis=0)
X_max = X.max(axis=0)

X_norm = (X - X_min) / (X_max - X_min + 1e-8)

# Target
Y_min = Y.min(axis=0)
Y_max = Y.max(axis=0)

# Scale to [0, 1]
Y_01 = (Y - Y_min) / (Y_max - Y_min + 1e-8)
# Compress into the [0.1, 0.9]
Y_norm = Y_01 * 0.8 + 0.1

print("X_norm range:", X_norm.min(), "to", X_norm.max())
print("Y_norm range:", Y_norm.min(), "to", Y_norm.max())

# TASK 2: TRAINING WORKFLOW
print("\n--- Task 2: Single Split & Plot ---")

# parameters
h = 12
n_restarts = 5
epochs = 50
test_split = 0.2

# split data
n = X_norm.shape[0]
n_test = int(n * test_split)
X_train = X_norm[:-n_test]
Y_train = Y_norm[:-n_test]
X_test = X_norm[-n_test:]
Y_test = Y_norm[-n_test:]

# store results
all_histories = []
best_mse = float('inf')

print(f"Training h={h} ...")

# multi-start loop
for i in range(n_restarts):
    print(f"  > Run {i+1}/{n_restarts}")
    
    # create model
    model = tf_keras.Sequential()
    model.add(tf_keras.layers.Dense(h, activation='sigmoid', input_shape=(16,)))
    model.add(tf_keras.layers.Dense(2, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse')
    
    # train
    history = model.fit(X_train, Y_train, epochs=epochs, verbose=0, batch_size=32)
    all_histories.append(history.history['loss'])
    
    # test
    current_mse = model.evaluate(X_test, Y_test, verbose=0)
    
    # keep best
    if current_mse < best_mse:
        best_mse = current_mse

print(f"Best MSE: {best_mse:.5f}")

# plot histories
plt.figure(figsize=(10, 6))
for i, loss_curve in enumerate(all_histories):
    plt.plot(loss_curve, label=f'Run {i+1}')

plt.title(f"Loss History (h={h})")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()


# TASK 3: MODEL SELECTION
print("\n--- Task 3: Cross-Validation ---")

# parameters for selection
h_list = [2, 4, 8, 16]
k_folds = 5
n_restarts_t3 = 3
epochs_t3 = 40

# store final stats
results = {} 

n = X_norm.shape[0]
fold_size = n // k_folds

# outer loop (h values)
for h_val in h_list:
    print(f"\nTesting h = {h_val}")
    
    fold_mses = []
    
    # k-fold loop
    for k in range(k_folds):
        # indices
        start = k * fold_size
        end = (k + 1) * fold_size
        
        # split validation and training
        X_val_k = X_norm[start:end]
        Y_val_k = Y_norm[start:end]
        
        X_train_k = np.concatenate((X_norm[:start], X_norm[end:]), axis=0)
        Y_train_k = np.concatenate((Y_norm[:start], Y_norm[end:]), axis=0)
        
        best_run_mse = float('inf')
        
        # inner multi-start loop
        for r in range(n_restarts_t3):
            # create model
            model = tf_keras.Sequential()
            model.add(tf_keras.layers.Dense(h_val, activation='sigmoid', input_shape=(16,)))
            model.add(tf_keras.layers.Dense(2, activation='linear'))
            model.compile(optimizer='adam', loss='mse')
            
            # train
            model.fit(X_train_k, Y_train_k, epochs=epochs_t3, verbose=0, batch_size=32)
            
            # validate
            val_mse = model.evaluate(X_val_k, Y_val_k, verbose=0)
            
            if val_mse < best_run_mse:
                best_run_mse = val_mse
        
        fold_mses.append(best_run_mse)
        print(f"  Fold {k+1} MSE: {best_run_mse:.5f}")

    # calculate stats
    scores = np.array(fold_mses)
    median = np.percentile(scores, 50)
    p25 = np.percentile(scores, 25)
    p75 = np.percentile(scores, 75)
    
    # save stats
    results[h_val] = {
        'median': median,
        'spread': p75 - p25,
        'ratio': (p75 - p25) / (median + 1e-8)
    }

# print table
print("\n--- Final Results ---")
print(f"{'h':<5} {'Median':<15} {'Spread':<20} {'Ratio':<10}")

for h_val in h_list:
    s = results[h_val]
    print(f"{h_val:<5} {s['median']:.5f}           {s['spread']:.5f}               {s['ratio']:.3f}")