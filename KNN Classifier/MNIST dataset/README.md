###  KNN - MNIST README

```markdown
# KNN Classifier - MNIST Digits

This project applies the K-Nearest Neighbors (KNN) algorithm to recognize handwritten digits from the MNIST dataset.

## Files
* `KnnClassifier.py`: Custom implementation of the KNN algorithm.
* `DigitsTest.py`: Main script to run experiments.

## Methodology
* **Data:** Uses the MNIST dataset, normalized to a 0-1 range.
* **Experiment:** Tests multiple values of `k` (e.g., 1, 3, 5, 10).
* **Validation:** Runs multiple random train/test splits for each `k` to calculate the mean accuracy and standard deviation (spread).

## Usage
Run the test script:
```bash
python DigitsTest.py