### Naive Bayes - Weather README

```markdown
# Weather Prediction (Play/No Play)

A simple classification project predicting whether to play sports based on weather conditions using Naive Bayes.

## Files
* `WeatherPrediction.py`: Script for preprocessing and classification.
* `weather.csv`: Dataset containing weather outlook, temperature, humidity, and wind conditions.

## Methodology
* **Preprocessing:** Encodes categorical text data into numerical format using `LabelEncoder`.
* **Model:** Trains a Categorical Naive Bayes model on the processed features.
* **Evaluation:** Reports accuracy and visualizes the confusion matrix.

## Usage
Run the prediction script:
```bash
python WeatherPrediction.py