# KNN Classifier - Wine Dataset

This project uses KNN to classify wine varieties based on chemical analysis features.

## Files
* `KnnClassifier.py`: Custom implementation of the KNN algorithm.
* `WineTest.py`: Main script for data processing, training, and evaluation.

## Methodology
* **Preprocessing:** Features are scaled using Min-Max Normalization to ensure distance calculations are accurate.
* **Evaluation:** Computes Accuracy, Precision, Recall, and F1-Score.
* **Visualization:** Uses Principal Component Analysis (PCA) to project the data into 2D and visualize decision boundaries.

## Usage
Run the test script:
```bash
python WineTest.py
```