# 03 · Optimization Algorithms & Loss Surface Dynamics

> **Core Focus:** Comparative study of first-order gradient descent variants and adaptive learning rate optimizers (SGD, SGD with Momentum, RMSprop, Adam) on deep feedforward networks.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-deep-learning-labs/blob/main/03_optimization_algorithms/01_mnist_optimizers_benchmark.ipynb)

---

## ⚙️ Mathematical Formulations

### 1. Stochastic Gradient Descent (SGD)
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$
* **Mechanics:** Updates parameters in the direction of the negative batch gradient.
* **Limitation:** Highly vulnerable to local minima, saddle points, and oscillatory behavior along steep ravine surfaces.

### 2. SGD with Polyak Momentum
$$v_{t+1} = \beta v_t + \eta \nabla_\theta \mathcal{L}(\theta_t), \quad \theta_{t+1} = \theta_t - v_{t+1}$$
* **Mechanics:** Accumulates an exponentially decaying velocity vector $v_t$ along dimensions of consistent gradient direction ($\beta \approx 0.9$).
* **Advantage:** Damps oscillations in orthogonal directions and accelerates through flat plateau surfaces.

### 3. RMSprop (Root Mean Square Propagation)
$$s_{t+1} = \gamma s_t + (1 - \gamma) (\nabla_\theta \mathcal{L}(\theta_t))^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_{t+1} + \epsilon}} \nabla_\theta \mathcal{L}(\theta_t)$$
* **Mechanics:** Maintains an exponentially weighted moving average of squared past gradients ($s_t$), scaling the learning rate per parameter inversely by the gradient magnitude.
* **Advantage:** Overcomes aggressive learning rate decay in non-stationary problems.

### 4. Adam (Adaptive Moment Estimation)
Combines first-order momentum (first moment $m_t$) and RMSprop adaptive variance scaling (second uncentered moment $v_t$) with bias correction:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

*(Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$).*

---

## 🧪 Benchmark Setup

* **Dataset:** MNIST Digits ($28 \times 28$ pixels, 60k train / 10k test).
* **Base Architecture:**
  $$\text{Input}(784) \longrightarrow \text{Dense}(128, \text{ReLU}) \longrightarrow \text{Dense}(64, \text{ReLU}) \longrightarrow \text{Dense}(10, \text{Softmax})$$
* **Controlled Variables:** Identical initialization seed, batch size ($64$), loss function (Categorical Cross-Entropy), and total training epochs ($20$).
* **Measured Responses:**
  * Training Loss Trajectory
  * Validation Loss Stability
  * Training & Validation Accuracy Progression
  * Epochs required to reach $\ge 95\%$ test accuracy

---

## 📊 Empirical Performance Matrix

| Optimizer | Adaptive Learning Rate? | Momentum Term? | Convergence Speed | Final Val Loss | Final Val Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard SGD** | ❌ No | ❌ No | Slow ($\sim 18$ epochs) | $0.214$ | $94.2\%$ |
| **SGD + Momentum**| ❌ No | ✅ Yes ($\beta=0.9$) | Moderate ($\sim 11$ epochs)| $0.138$ | $96.8\%$ |
| **RMSprop** | ✅ Yes | ❌ No | Fast ($\sim 7$ epochs) | $0.098$ | $97.6\%$ |
| **Adam** | ✅ Yes | ✅ Yes | **Fastest ($\sim 5$ epochs)** | **$0.082$** | **$98.2\%$** |

---

## 📌 Architectural Insights
1. **SGD vs. Momentum:** Adding momentum ($\beta = 0.9$) substantially speeds up initial loss drop and reduces cross-iteration variance.
2. **Adaptive Domination:** Both RMSprop and Adam outperform vanilla SGD by over **3.5 percentage points** in final accuracy, with Adam delivering the smoothest validation curve without premature plateauing.
