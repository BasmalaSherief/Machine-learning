# MNIST Autoencoder & Image Restoration

This project implements an Autoencoder using TensorFlow/Keras to perform unsupervised learning tasks on the MNIST dataset.

## Features
The script `Autoencoder.py` performs three main tasks:

1.  **Data Filtering:** Selects specific digit classes (e.g., 5 and 7) from the MNIST dataset for focused analysis.
2.  **Image Embedding:** Trains a shallow autoencoder to compress 784-pixel images into a 2D latent space, visualizing how the network separates classes without labels.
3.  **Image Restoration:** Trains a denoising autoencoder to reconstruct original images from versions masked with random stripes and blocks.

## Usage
Run the script to execute the pipeline:
```bash
python Autoencoder.py
```

## Results

* **2D Embedding:** A scatter plot showing the separation of digit classes in the latent space.

* **Restoration:** A visualization comparing Original, Masked, and Restored images to evaluate reconstruction quality.