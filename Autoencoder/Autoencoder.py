import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tf_keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.datasets import mnist

# Load data
(X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

# Data processing
class1 = 5
class2 = 7

# Filter the data
def filter_by_class(X, y, c1, c2):
    mask = (y == c1) | (y == c2)
    X_filtered = X[mask]
    y_filtered = y[mask]
    return X_filtered, y_filtered

# Apply the filter to datasets
X_train, y_train = filter_by_class(X_train_full, y_train_full, class1, class2)
X_test, y_test = filter_by_class(X_test_full, y_test_full, class1, class2)

# Data normalization
# MNIST pixels are 0-255, scale them to 0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten the images
# 28x28 images become 784-dimensional vectors
X_train = X_train.reshape((X_train.shape[0], 784))
X_test = X_test.reshape((X_test.shape[0], 784))

# Verification
print(f"Selected classes: {class1} and {class2}")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")


# Hyperparameters
input_dim = 784
hidden_dim = 2   
epochs = 50
batch_size = 128
n_restarts = 5   # Multi-start to ensure we find a good minimum

best_loss = float('inf')
best_autoencoder = None
best_encoder = None
best_history = None

for i in range(n_restarts):
    print(f"Training Run {i+1}/{n_restarts}...")
    
    # Input Layer
    input_img = Input(shape=(input_dim,))
    
    # Hidden Layer (The Encoder)
    encoded = Dense(hidden_dim, 
                    activation='sigmoid', 
                    kernel_initializer=RandomUniform(minval=-0.7, maxval=0.7))(input_img)
    
    # Output Layer (The Decoder)
    decoded = Dense(input_dim, activation='linear')(encoded)
    
    # The Full Autoencoder (Input -> Output) for training
    autoencoder = Model(inputs=input_img, outputs=decoded)
    
    # The Encoder (Input -> Hidden) for plotting later
    encoder = Model(inputs=input_img, outputs=encoded)
    
    # Compile
    opt = tf_keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=opt, loss='mse')
    
    # Train
    history = autoencoder.fit(X_train, X_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              verbose=0) 
    
    final_loss = history.history['loss'][-1]
    print(f"  > Final MSE: {final_loss:.5f}")
    
    if final_loss < best_loss:
        best_loss = final_loss
        best_autoencoder = autoencoder
        best_encoder = encoder
        best_history = history

print(f"\nBest MSE achieved: {best_loss:.5f}")


# Plotting
print("\n--- Visualizing the Latent Space ---")

latent_points = best_encoder.predict(X_test)

# Plot
plt.figure(figsize=(10, 8))

# Scatter plot: x = hidden_unit_1, y= hidden_unit_2
scatter = plt.scatter(latent_points[:, 0], latent_points[:, 1], c=y_test, cmap='viridis', alpha=0.7)

plt.colorbar(scatter, label='Digit Class')
plt.title(f'2D Embedding of Digits {class1} and {class2} (MSE: {best_loss:.4f})')
plt.xlabel('Hidden Unit 1 (Sigmoid)')
plt.ylabel('Hidden Unit 2 (Sigmoid)')
plt.grid(True)
plt.show()


print("\n" + "="*40)
print("--- TASK 3: Image Restoration ---")
print("="*40)

def apply_random_masks(X, img_shape=(28, 28)):
    
    X_masked = X.copy()
    n_samples = X.shape[0]
    X_reshaped = X_masked.reshape((n_samples, img_shape[0], img_shape[1]))
    
    for i in range(n_samples):
        mask_type = np.random.randint(0, 3)
        
        if mask_type == 0: # Random Rectangular Patch 
            # Size of the block
            h = np.random.randint(5, 15) # Height 5 to 15 pixels
            w = np.random.randint(5, 15) # Width 5 to 15 pixels
            # Top-left corner
            y = np.random.randint(0, img_shape[0] - h)
            x = np.random.randint(0, img_shape[1] - w)
            # Apply mask 
            X_reshaped[i, y:y+h, x:x+w] = 0.0
            
        elif mask_type == 1: # Horizontal Stripes
            # Mask every other 2 or 3 rows
            step = np.random.randint(2, 5)
            for r in range(0, img_shape[0], step):
                X_reshaped[i, r:r+1, :] = 0.0
                
        elif mask_type == 2: # Vertical Stripes
            # Mask random columns
            step = np.random.randint(2, 5)
            for c in range(0, img_shape[1], step):
                X_reshaped[i, :, c:c+1] = 0.0

    # Flatten back to (N, 784)
    return X_reshaped.reshape((n_samples, -1))

# New Data Sets
print("Generating masked images...")
X_train_masked = apply_random_masks(X_train)
X_test_masked = apply_random_masks(X_test)

# Restoration Network
h_restoration = 64  # Increased from 2 to 64

restore_input = Input(shape=(784,))
restore_encoded = Dense(h_restoration, activation='sigmoid')(restore_input)
restore_decoded = Dense(784, activation='linear')(restore_encoded)

restorer = Model(inputs=restore_input, outputs=restore_decoded)

restorer.compile(optimizer='adam', loss='mse')

# Train
print(f"Training restoration model with h={h_restoration}...")
restorer.fit(X_train_masked, X_train, 
             epochs=30, 
             batch_size=128, 
             shuffle=True, 
             verbose=0)

# Evaluation & Visualization
print("Visualizing restoration results...")
predicted_images = restorer.predict(X_test_masked)

# Plotting
n_show = 8
plt.figure(figsize=(16, 6))
indices = np.random.choice(len(X_test), n_show, replace=False)

for i, idx in enumerate(indices):
    # Original
    ax = plt.subplot(3, n_show, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis("off")
    
    # Masked 
    ax = plt.subplot(3, n_show, i + 1 + n_show)
    plt.imshow(X_test_masked[idx].reshape(28, 28), cmap='gray')
    plt.title("Masked")
    plt.axis("off")
    
    # Restored 
    ax = plt.subplot(3, n_show, i + 1 + 2*n_show)
    plt.imshow(predicted_images[idx].reshape(28, 28), cmap='gray')
    plt.title("Restored")
    plt.axis("off")

plt.tight_layout()
plt.show()
