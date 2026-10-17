# 🧠 Applied Deep Learning Labs

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Math](https://img.shields.io/badge/Math-NumPy%20%7C%20SciPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![Visualization](https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20Seaborn-232F3E?style=flat)](https://matplotlib.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A rigorous laboratory suite covering fundamental and modern deep learning paradigms: Feedforward Multi-Layer Perceptrons (MLPs), mathematical non-linearity analysis, adaptive optimization dynamics, and Convolutional Neural Networks (CNNs) for computer vision.

---

## 🗺️ Curriculum & Laboratory Modules

```text
applied-deep-learning-labs/
├── 01_feedforward_neural_networks/     # MLPs for continuous regression & binary classification
├── 02_activation_functions/            # Mathematical derivatives & empirical MNIST saturation analysis
├── 03_optimization_algorithms/         # SGD, Momentum, RMSprop, Adam loss surface dynamics
└── 04_convolutional_neural_networks/   # Hierarchical spatial convolutions on CIFAR-10
```

---

## 🔬 Experiment Index & Benchmark Suite

| Module | Notebook | Dataset | Architecture | Primary Metric | Colab Link |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **01 · Feedforward Networks** | `01_california_housing_regression.ipynb` | California Housing | Dense 64 $\to$ 32 $\to$ 16 $\to$ 1 | MAE / $R^2$ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-deep-learning-labs/blob/main/01_feedforward_neural_networks/01_california_housing_regression.ipynb) |
| **01 · Feedforward Networks** | `02_breast_cancer_classification.ipynb` | Wisconsin Breast Cancer | Dense 32 $\to$ 16 $\to$ 1 (Sigmoid) | $F_1 = 0.98$ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-deep-learning-labs/blob/main/01_feedforward_neural_networks/02_breast_cancer_classification.ipynb) |
| **02 · Activation Functions** | `01_mathematical_derivatives_analysis.ipynb` | Synthetic $x \in [-10, 10]$ | Vectorized NumPy math analysis | Analytical $\sigma', \tanh', \text{ReLU}'$ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-deep-learning-labs/blob/main/02_activation_functions/01_mathematical_derivatives_analysis.ipynb) |
| **02 · Activation Functions** | `02_mnist_activation_benchmark.ipynb` | MNIST Digits | Dense 128 $\to$ 64 $\to$ 10 | Accuracy (ReLU: $98.1\%$) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-deep-learning-labs/blob/main/02_activation_functions/02_mnist_activation_benchmark.ipynb) |
| **03 · Optimization Dynamics** | `01_mnist_optimizers_benchmark.ipynb` | MNIST Digits | Dense 128 $\to$ 64 $\to$ 10 | Adam vs. RMSprop vs. SGD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-deep-learning-labs/blob/main/03_optimization_algorithms/01_mnist_optimizers_benchmark.ipynb) |
| **04 · Convolutional Vision** | `01_cifar10_cnn_classification.ipynb` | CIFAR-10 ($32\times32$ RGB) | Conv2D 32 $\to$ 64 $\to$ 128 $\to$ Dense 128 | Top-1 Accuracy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-deep-learning-labs/blob/main/04_convolutional_neural_networks/01_cifar10_cnn_classification.ipynb) |

---

## 📊 Key Findings & Comparative Insights

### 1. Vanishing Gradient Mitigation
* **Sigmoid** derivatives peak at $\sigma'(0) = 0.25$, creating exponential gradient attenuation across stacked layers ($\prod \le 0.25^L$).
* **$\tanh$** zero-centers activations with peak gradient $1.0$, improving convergence rate over Sigmoid by $\sim 1.5\times$.
* **$\text{ReLU}$** maintains a constant non-saturating gradient of $1.0$ for all positive pre-activations, delivering the fastest training convergence and lowest cross-entropy loss.

### 2. Optimizer Landscape
* **Stochastic Gradient Descent (SGD):** Suffers from high gradient variance and slow valley navigation.
* **Polyak Momentum:** Dampens oscillatory orthogonal vectors ($\beta = 0.9$), yielding a **$2.6\%$ accuracy boost** over vanilla SGD.
* **Adam:** Combines first-moment velocity with second-moment adaptive variance scaling, reaching optimal convergence in just **5 epochs** compared to 18 epochs for vanilla SGD.

### 3. Spatial Convolution vs. Dense Feature Flattening
* By preserving 2D spatial locality and enforcing parameter sharing, the **Conv2D hierarchical pipeline** extracts edge, contour, and object-level textures from raw $32\times 32\times 3$ RGB inputs without the astronomical parameter bloat of fully connected layers.

---

## 🚀 Quickstart & Reproduction

### Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment (`venv` or `conda`)

### 1. Clone the Repository
```bash
git clone https://github.com/Karthick-564/applied-deep-learning-labs.git
cd applied-deep-learning-labs
```

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch Notebooks
```bash
jupyter lab
```

Or open any notebook directly in Google Colab using the badges in the [Experiment Index](#-experiment-index--benchmark-suite).

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).