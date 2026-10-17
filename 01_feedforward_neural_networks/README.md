# 01 · Feedforward Neural Networks (MLP)

> **Core Focus:** Foundations of Multi-Layer Perceptrons (MLPs), Forward Propagation, Backpropagation with Gradient Descent, Continuous Regression, and Binary Classification.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-deep-learning-labs/blob/main/01_feedforward_neural_networks/01_california_housing_regression.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-deep-learning-labs/blob/main/01_feedforward_neural_networks/02_breast_cancer_classification.ipynb)

---

## 🔬 Theoretical Foundations

A Multi-Layer Perceptron is a feedforward artificial neural network mapping an input feature vector $\mathbf{x} \in \mathbb{R}^d$ to an output prediction $\hat{\mathbf{y}}$ through stacked affine transformations parameterized by weight matrices $\mathbf{W}^{(l)}$ and bias vectors $\mathbf{b}^{(l)}$, interleaved with non-linear activation functions $\sigma(\cdot)$:

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \quad \mathbf{a}^{(l)} = \sigma\left(\mathbf{z}^{(l)}\right)$$

Where:
* $\mathbf{a}^{(0)} = \mathbf{x}$ is the raw input vector.
* Layer $l \in \{1, 2, \dots, L\}$, where $L$ is the final output layer.

---

## 📂 Notebook Implementations

### 1. `01_california_housing_regression.ipynb`
* **Task:** Predicting median house values based on spatial, demographic, and socioeconomic indicators.
* **Dataset:** California Housing ($N = 20,640$ samples, 8 continuous features).
* **Preprocessing:** Z-score normalization via `StandardScaler()`:
  $$z = \frac{x - \mu}{\sigma}$$
* **Network Topology:**
  * Input Layer: 8 features
  * Hidden Layer 1: 64 units, ReLU activation
  * Hidden Layer 2: 32 units, ReLU activation
  * Hidden Layer 3: 16 units, ReLU activation
  * Output Layer: 1 linear unit ($\hat{y} \in \mathbb{R}$)
* **Objective Function:** Mean Squared Error (MSE):
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N \left(y_i - \hat{y}_i\right)^2$$
* **Key Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), $R^2$ score.

---

### 2. `02_breast_cancer_classification.ipynb`
* **Task:** Binary diagnostic classification (Malignant vs. Benign).
* **Dataset:** Wisconsin Breast Cancer ($N = 569$ samples, 30 digitized cell nuclei morphological features).
* **Preprocessing:** Train-test split (80/20) + Feature standardization.
* **Network Topology:**
  * Input Layer: 30 features
  * Hidden Layer 1: 32 units, ReLU activation
  * Hidden Layer 2: 16 units, ReLU activation
  * Output Layer: 1 unit, Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$
* **Objective Function:** Binary Cross-Entropy Loss:
  $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
* **Evaluation:** Confusion Matrix, Precision, Recall, Specificity, and $F_1$-score.

---

## 📊 Summary Comparison

| Pipeline | Target Task | Input Dim | Hidden Layers | Output Activation | Loss Function | Primary Metric |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Housing Model** | Regression | 8 | [64, 32, 16] | Linear | Mean Squared Error (MSE) | MAE / $R^2$ |
| **Diagnostics Model**| Binary Classification | 30 | [32, 16] | Sigmoid | Binary Cross-Entropy | Precision / Recall / $F_1$ |

---

## 🚀 Execution Guide

```bash
# Clone the repository
git clone https://github.com/Karthick-564/applied-deep-learning-labs.git
cd applied-deep-learning-labs/01_feedforward_neural_networks

# Launch in Jupyter Notebook / Lab
jupyter lab 01_california_housing_regression.ipynb
```
