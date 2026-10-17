# 04 · Convolutional Neural Networks (CNN) for Computer Vision

> **Core Focus:** Hierarchical spatial feature representation, 2D convolution operations, weight sharing, receptive field expansion, and multi-class image classification on the CIFAR-10 benchmark.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthick-564/applied-deep-learning-labs/blob/main/04_convolutional_neural_networks/01_cifar10_cnn_classification.ipynb)

---

## 👁️ Convolution Mechanics & Spatial Hierarchy

Unlike traditional Dense layers that collapse 2D image geometry into 1D vectors (destroying spatial correlation and causing parameter explosion), Convolutional Neural Networks exploit two fundamental physical principles:
1. **Local Connectivity (Receptive Fields):** Neurons in a convolutional layer process only a localized patch of the input volume.
2. **Translation Equivariance via Parameter Sharing:** The same learned filter kernel $\mathbf{K} \in \mathbb{R}^{k_h \times k_w \times c_{\text{in}}}$ slides across the entire spatial dimension:

$$S(i, j) = (\mathbf{I} * \mathbf{K})(i, j) = \sum_{m} \sum_{n} \sum_{c} \mathbf{I}(i+m, j+n, c) \cdot \mathbf{K}(m, n, c) + b$$

---

## 🏗️ Model Architecture Pipeline

The network processes 3-channel RGB color images through an expanding feature hierarchy:

```text
Input: (32 x 32 x 3)
   │
   ▼
Conv2D (32 filters, 3x3, same padding, ReLU)
   │  --> Output: (32 x 32 x 32) [Low-level: Edges, gradients, color contrasts]
   ▼
Conv2D (64 filters, 3x3, same padding, ReLU)
   │  --> Output: (32 x 32 x 64) [Mid-level: Textures, simple shapes, contours]
   ▼
Conv2D (128 filters, 3x3, same padding, ReLU)
   │  --> Output: (32 x 32 x 128) [High-level: Structural parts, object components]
   ▼
Flatten Layer
   │  --> Output: Vector (131,072 dimensions)
   ▼
Dense Layer (128 units, ReLU activation)
   │  --> Feature compression & dense non-linear mapping
   ▼
Dense Output Head (10 units, Softmax activation)
   │  --> Class probabilities across 10 CIFAR-10 categories
```

---

## 🎨 Dataset Profile: CIFAR-10

* **Source:** Canadian Institute for Advanced Research (CIFAR-10).
* **Total Volume:** 60,000 $32 \times 32$ RGB natural color images.
* **Split:** 50,000 Training images / 10,000 Test images.
* **Classes (10):**
  * ✈️ Airplane, 🚗 Automobile, 🐦 Bird, 🐱 Cat, 🦌 Deer
  * 🐶 Dog, 🐸 Frog, 🐴 Horse, 🚢 Ship, 🚚 Truck
* **Preprocessing:** Pixel value scaling from $[0, 255]$ into floating-point normalized range $[0.0, 1.0]$.

---

## 📊 Training Specifications & Performance

| Parameter | Configuration |
| :--- | :--- |
| **Loss Function** | Sparse Categorical Cross-Entropy |
| **Optimizer** | Adam ($\eta = 0.001$) |
| **Batch Size** | 64 |
| **Validation Strategy** | Holdout test set evaluation |
| **Hardware Acceleration** | GPU (NVIDIA CUDA / Google Colab T4) |

### Key Results
* **Spatial Feature Extraction:** Successive convolutional blocks capture increasingly abstract geometric representations without manual feature engineering.
* **Regularization Opportunities:** Demonstrated how deep convolutional stacks benefit from Batch Normalization and Spatial Dropout to mitigate validation overfitting.

---

## 🚀 Execution Guide

```bash
cd applied-deep-learning-labs/04_convolutional_neural_networks
jupyter lab 01_cifar10_cnn_classification.ipynb
```
