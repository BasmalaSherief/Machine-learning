# Turbine Health Estimation

This project uses Neural Network Regression to predict the health decay coefficients of Gas Turbine components (Compressor and Turbine) based on sensor measurements.

## Files
* `TurbineHealthWorkflow.py`: Implements the complete training pipeline, including data loading, normalization, and multi-start optimization to avoid local minima.
* `TH_ModelSelection.py`: Performs model selection using k-Fold Cross-Validation to determine the optimal number of hidden units ($h$).
* **UCI CBM Dataset:** Contains the `data.txt` used for training.

## Methodology
* **Task:** Multi-output regression (16 inputs -> 2 outputs).
* **Normalization:** Inputs are Min-Max scaled; Targets are scaled to a safe range [0.1, 0.9] for sigmoid activation.
* **Model Selection:** Evaluates different network sizes (values of $h$) to balance bias (Median MSE) and variance (Spread).

## Usage
To run the standard training workflow:
```bash
python TurbineHealthWorkflow.py
```

To run model selection (cross-validation):

```Bash
python TH_ModelSelection.py
```