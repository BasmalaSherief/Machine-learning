# KNN Classifier - MNIST Digit Recognition

Custom implementation of K-Nearest Neighbors for handwritten digit classification on the MNIST dataset.

## 📋 Overview

This experiment applies the K-Nearest Neighbors algorithm to classify handwritten digits (0-9) from the MNIST database. The focus is on understanding how the hyperparameter **k** affects classification performance in high-dimensional image data.

---

## 🎯 Dataset: MNIST

**Description**: The Modified National Institute of Standards and Technology database of handwritten digits.

**Statistics**:
- **Training Samples**: 60,000 images
- **Test Samples**: 10,000 images
- **Image Dimensions**: 28×28 pixels (grayscale)
- **Features**: 784 (flattened pixel values)
- **Classes**: 10 (digits 0-9)
- **Class Distribution**: Relatively balanced (~6,000-7,000 samples per digit)

**Preprocessing**:
- Pixel values normalized to [0, 1] by dividing by 255
- No additional feature engineering required
- Images flattened from 28×28 matrices to 784-dimensional vectors

---

## 🔬 Experiment: Effect of k on Classification Accuracy

### Objective
Determine the optimal value of k for digit classification by analyzing the bias-variance trade-off.

### Methodology

**Experimental Design**:
- Test values: k ∈ {1, 3, 5, 10}
- **30 independent trials** per k value (with random subsampling)
- Compute mean accuracy and standard deviation
- Visualize results with error bars

**Why Multiple Trials?**
- Accounts for sampling variability
- Provides confidence intervals
- Reveals stability of each k value

### Distance Metric
Euclidean distance in 784-dimensional space:
```
d(x, y) = √(Σ(x_i - y_i)²)
```

---

## 📊 Results

### Performance Table

| k | Mean Accuracy | Std Dev | Interpretation |
|---|---------------|---------|----------------|
| 1 | 0.9469 | 0.0050 | High variance, sensitive to outliers |
| **3** | **0.9499** | **0.0047** | **Optimal: Best accuracy, low variance** |
| 5 | 0.9479 | 0.0047 | Slight over-smoothing of boundaries |
| 10 | 0.9434 | 0.0043 | Too much smoothing, loses local structure |

**Best Performance**: k=3 achieves **94.99% accuracy**

### Visualization

![MNIST k Analysis](mnist_k_plot.png)  
*Mean accuracy vs. k across 30 trials. Error bars represent ±1 standard deviation.*

---

## 🔍 Analysis

### Why k=3 is Optimal

1. **Local Structure**: MNIST digits have clear class boundaries; small k captures fine-grained patterns
2. **Noise Robustness**: k=3 provides some averaging (better than k=1) without over-smoothing
3. **Computational Efficiency**: Smaller k = faster predictions
4. **Stability**: Low standard deviation (0.47%) indicates consistent performance

### Why Larger k Values Perform Worse

**k=1 (Nearest Neighbor)**:
- Too sensitive to outliers and mislabeled training examples
- No averaging to reduce noise
- Slightly lower accuracy than k=3

**k=10 (Over-smoothing)**:
- Decision boundaries become too broad
- Minority class patterns get overwhelmed by nearby majority classes
- Example: A "1" near several "7"s might be misclassified

### High-Dimensional Considerations

**Curse of Dimensionality**: In 784-dimensional space, distances become less meaningful as dimensionality increases. However:
- MNIST images lie on a lower-dimensional manifold (digit structure)
- Euclidean distance still captures similarity well for this data
- Normalization is critical to ensure all pixels contribute equally

---

## 📈 Performance Breakdown by Digit

Common confusion patterns observed:

| Digit Pair | Confusion Rate | Reason |
|------------|----------------|--------|
| 4 ↔ 9 | Moderate | Similar top loops |
| 3 ↔ 5 | Moderate | Overlapping curves |
| 7 ↔ 9 | Low-Moderate | Similar vertical strokes |
| 5 ↔ 8 | Low | Writing style variations |

**Most Accurate**: Digits 0 and 1 (distinct shapes)  
**Most Challenging**: Digits 4, 7, 9 (structural similarities)

---

## 🚀 Key Findings

### Takeaways

1. **Low k is Preferred**: For MNIST, k ∈ {3, 5} provides the best performance
2. **Stability**: Low variance across trials confirms robust predictions
3. **No Feature Engineering Needed**: Raw pixels + normalization is sufficient
4. **Computational Trade-off**: 94.99% accuracy comes at the cost of storing 60,000 training samples

### Comparison to Other Methods

| Method | Accuracy | Training Time | Prediction Time |
|--------|----------|---------------|-----------------|
| **KNN (k=3)** | **94.99%** | Instant | Slow (O(n×d)) |
| Naive Bayes | ~84% | Fast | Fast |
| Neural Network | 97-99% | Slow | Fast |
| CNN | >99% | Very Slow | Fast |

**KNN Advantage**: Simple, interpretable, no training phase  
**KNN Limitation**: Slow predictions, memory intensive

---

## 🛠️ Running the Experiment

**Requirements**:
```bash
pip install numpy scikit-learn matplotlib
```

**Execute**:
```bash
python knn_mnist.py
```

**Output**:
- Accuracy statistics for each k value
- Visualization saved as `mnist_k_plot.png`
- CSV report with detailed trial results

---