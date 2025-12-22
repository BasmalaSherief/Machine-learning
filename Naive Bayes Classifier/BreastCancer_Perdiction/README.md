###  Naive Bayes - Breast Cancer README

```markdown
# Breast Cancer Recurrence Prediction

This project uses a Categorical Naive Bayes classifier to predict recurrence events in breast cancer patients.

## Files
* `BreastCancerPrediction.py`: Script containing the classifier class and evaluation logic.
* `BreastCancer.csv`: Dataset containing patient attributes.

## Methodology
* **Data Handling:** Fills missing values and encodes categorical/ordinal features (e.g., tumor-size, age ranges).
* **Model:** Uses `CategoricalNB` from Scikit-learn.
* **Evaluation:** Outputs Accuracy, Classification Report, and a Confusion Matrix plot.

## Usage
Run the prediction script:
```bash
python BreastCancerPrediction.py