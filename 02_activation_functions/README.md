# 02 · Activation Functions & Non-Linearity Dynamics

> **Core Focus:** Mathematical derivations, first-order derivative behavior, vanishing gradient phenomena, and empirical benchmarking on the MNIST handwritten digit classification task.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-deep-learning-labs/blob/main/02_activation_functions/01_mathematical_derivatives_analysis.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-deep-learning-labs/blob/main/02_activation_functions/02_mnist_activation_benchmark.ipynb)

---

## 📐 Mathematical Formulations & Derivatives

Activation functions introduce non-linearity into neural networks, enabling them to approximate complex, non-convex decision boundaries (Universal Approximation Theorem).

### 1. Logistic Sigmoid
$$\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \frac{d\sigma(x)}{dx} = \sigma(x)(1 - \sigma(x))$$
* **Range:** $(0, 1)$
* **Maximum Derivative:** $\sigma'(0) = 0.25$
* **Limitation:** Non-zero centered outputs; induces severe vanishing gradients when $|x| \gg 0$.

### 2. Hyperbolic Tangent ($\tanh$)
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}, \quad \frac{d\tanh(x)}{dx} = 1 - \tanh^2(x)$$
* **Range:** $(-1, 1)$
* **Maximum Derivative:** $\tanh'(0) = 1.0$
* **Advantage:** Zero-centered output gradients; mitigates zig-zagging gradient dynamics compared to Sigmoid.

### 3. Rectified Linear Unit ($\text{ReLU}$)
$$\text{ReLU}(x) = \max(0, x), \quad \frac{d\text{ReLU}(x)}{dx} = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$$
* **Range:** $[0, \infty)$
* **Advantage:** Constant non-saturating gradient of $1.0$ for positive activations; completely prevents vanishing gradients in positive regime and promotes sparse representation.

---

## 🧪 Experiments Overview

### 1. `01_mathematical_derivatives_analysis.ipynb`
* **Implementation:** Vectorized NumPy implementations of $\sigma(x)$, $\tanh(x)$, $\text{ReLU}(x)$ and their respective analytical first derivatives.
* **Input Domain:** $x \in [-10, 10]$ discretized across 1,000 points.
* **Visualizations:** Dual-panel plots illustrating value curves alongside gradient attenuation profiles.
* **Analysis:** Demonstrates why backpropagation through multiple Sigmoid layers causes gradients to diminish exponentially as $\prod_{l} \sigma'(z^{(l)}) \le (0.25)^L \to 0$.

---

### 2. `02_mnist_activation_benchmark.ipynb`
* **Benchmark Task:** Multi-class classification on the MNIST handwritten digit dataset ($28 \times 28$ grayscale images, 10 classes, 60,000 train / 10,000 test).
* **Controlled Architecture:**
  $$\text{Input}(784) \longrightarrow \text{Dense}(128, \phi) \longrightarrow \text{Dense}(64, \phi) \longrightarrow \text{Dense}(10, \text{Softmax})$$
  *(All hyper-parameters: Adam optimizer, Categorical Cross-Entropy loss, batch size 64, fixed initial random seed).*
* **Evaluated $\phi$:** $\text{Sigmoid}$, $\tanh$, and $\text{ReLU}$.

---

## 📈 Empirical Benchmark Results

| Activation Function | Zero-Centered? | Saturation Regime | Convergence Epochs | Training Loss | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sigmoid** | ❌ No | Both sides ($x \to \pm\infty$) | Slow ($\sim 15$ epochs) | Moderate | $\approx 96.5\%$ |
| **$\tanh$** | ✅ Yes | Both sides ($x \to \pm\infty$) | Moderate ($\sim 10$ epochs) | Low | $\approx 97.4\%$ |
| **$\text{ReLU}$** | ❌ Half-space | Negative side only ($x < 0$) | **Fastest ($\sim 6$ epochs)** | **Lowest** | **$\approx 98.1\%$** |

---

## 💡 Key Takeaways
1. **Convergence Speed:** $\text{ReLU}$ converges nearly $2.5\times$ faster than $\text{Sigmoid}$ due to non-saturating positive derivatives.
2. **Saturating Bounds:** For deep feedforward architectures, $\tanh$ consistently outperforms $\text{Sigmoid}$ due to zero-centered gradient normalization.
