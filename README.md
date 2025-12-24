# Shallow Autoencoder for MNIST Representation Learning (TensorFlow / Keras)

## Abstract

This repository presents a **shallow autoencoder framework** for unsupervised representation learning on the **MNIST handwritten digit dataset**. The objective is to learn a compact **32-dimensional latent embedding** from high-dimensional image data (784 pixels) and reconstruct the original input with minimal information loss.

The implementation emphasizes **conceptual clarity, interpretability, and methodological rigor**, making it suitable for **academic research portfolios, PhD applications, and foundational deep learning studies**.

---

## Research Motivation

Autoencoders form the backbone of many modern representation learning methods, including:
- Dimensionality reduction
- Feature extraction for downstream tasks
- Anomaly detection
- Pretraining for deep neural networks

This project demonstrates how even a **minimal, shallow autoencoder** can capture meaningful structure in image data when trained properly, serving as a strong baseline for more advanced architectures.

---

## Dataset

- **Dataset**: MNIST
- **Samples**:
  - 60,000 training images
  - 10,000 test images
- **Image Size**: 28 × 28 (grayscale)
- **Flattened Dimension**: 784

The dataset is loaded directly using TensorFlow/Keras utilities.

---

## Data Preprocessing

1. **Normalization**
   - Pixel values scaled to `[0, 1]` using float32 normalization.

2. **Train–Validation Split**
   - 10,000 samples reserved for validation.
   - Remaining samples used for training.

3. **Flattening**
   - Images reshaped from `(28, 28)` to `(784,)` vectors to match dense-layer input.

---

## Model Architecture

### Autoencoder
- Input Layer: 784 neurons
- Encoder Layer: 32 neurons (ReLU)
- Decoder Layer: 784 neurons (Sigmoid)

This symmetric structure enforces **information compression** while preserving reconstruction fidelity.

### Encoder
- Extracts 32-dimensional latent representation

### Decoder
- Reconstructs original image from latent space

---

## Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 256
- **Epochs**: Up to 50
- **Early Stopping**:
  - Monitors training loss
  - Prevents overfitting and unnecessary computation

---

## Evaluation Strategy

- Reconstruction loss evaluated on unseen test data
- Visualization of:
  - Original images
  - Latent representations reshaped for inspection
  - Reconstructed outputs
- Training and validation loss curves plotted for convergence analysis

---

## Latent Space Analysis

- Encoder output examined directly to understand compression behavior
- Latent vectors reshaped and visualized to inspect learned structure
- Decoder successfully reconstructs digit morphology from compact embeddings

---

## Reconstruction Analysis

- Side-by-side comparison of:
  - Original MNIST image
  - Autoencoder reconstruction
- Demonstrates effective preservation of digit identity despite heavy compression

---

## Confusion Matrix (Exploratory)

A confusion matrix is computed using argmax-based comparisons to explore reconstruction consistency.  
While autoencoders are unsupervised models, this analysis provides **diagnostic insight into pixel-dominant activations** and reconstruction alignment.

---

## Key Outcomes

- Successful compression from 784 → 32 dimensions
- Stable convergence with early stopping
- Clear reconstruction of handwritten digits
- Interpretable latent representations

---

## Research Significance

This project demonstrates:
- Foundational understanding of unsupervised learning
- Proper handling of training/validation splits
- Methodical evaluation beyond raw loss values
- Strong baseline suitable for extension to:
  - Sparse autoencoders
  - Variational autoencoders (VAEs)
  - Deep convolutional autoencoders

---

## Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Intended Use

This repository is intended for:
- Academic research
- Teaching and learning representation learning
- PhD and research portfolio demonstration

Not intended for clinical or production deployment without further validation.

---

## Author Note

This implementation reflects a **research-oriented approach** to neural representation learning, prioritizing **clarity, reproducibility, and theoretical grounding**.
