from pandas import read_csv
import numpy as np
import tensorflow
import tensorflow.keras as tf_keras
import matplotlib.pyplot as plt

# read the data
data = read_csv('/home/basmala/Windows-Educational/Robotics Master/ML/Assignments/NeuralNetworkRegression/condition+based+maintenance+of+naval+propulsion+plants/UCI CBM Dataset/data.txt', sep='\s+').values
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

# set up parameters
h = 12
n_restarts = 5        # How many times to retrain
epochs = 50           # Optimization duration
test_split = 0.2

# split the data
n = X_norm.shape[0]
n_test = int(n * test_split)
X_train = X_norm[:-n_test]
Y_train = Y_norm[:-n_test]
X_test = X_norm[-n_test:]
Y_test = Y_norm[-n_test:]

# store results
all_histories = []
best_mse = float('inf')

print(f"Training with h={h} ...")

for i in range(n_restarts):
    print(f"  > Restart {i+1}/{n_restarts}")
    
    # Build Model
    model = tf_keras.Sequential()
    model.add(tf_keras.layers.Dense(h, activation='sigmoid', input_shape=(16,))) # 16 inputs
    model.add(tf_keras.layers.Dense(2, activation='linear')) # 2 outputs 
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    history = model.fit(X_train, Y_train, epochs=epochs, verbose=0, batch_size=32)
    
    # Save history 
    all_histories.append(history.history['loss'])
    
    # Evaluate on Test Set
    current_mse = model.evaluate(X_test, Y_test, verbose=0)
    
    # Track Best Result
    if current_mse < best_mse:
        best_mse = current_mse

print(f"\nBest Test MSE found: {best_mse:.5f}")

# Visualization 
plt.figure(figsize=(10, 6))
for i, loss_curve in enumerate(all_histories):
    plt.plot(loss_curve, label=f'Run {i+1} (Final: {loss_curve[-1]:.4f})')

plt.title(f"Optimization Histories (h={h})")
plt.xlabel("Epochs")
plt.ylabel("Training MSE (Loss)")
plt.legend()
plt.grid(True)
plt.show()