# Fashion-MNIST Denoising Autoencoder

A **Deep Learning** project that removes Gaussian noise from grayscale images using a Convolutional Autoencoder built with TensorFlow/Keras.

---

## The Problem

Real-world images are often corrupted by noise — from sensor limitations, poor lighting, or lossy transmission. Traditional denoising filters (Gaussian blur, median filter) tend to destroy fine details. A learned approach can remove noise while preserving the structure of the image.

---

## Project Overview

This repository features a **Denoising Convolutional Autoencoder (CAE)** trained on the **Fashion-MNIST** dataset. The model learns the inverse mapping of noise, effectively restoring corrupted images to their clean originals.

1. **Loads Fashion-MNIST** from CSV files (28x28 grayscale images of clothing items)
2. **Corrupts images** with additive Gaussian noise (factor = 0.5)
3. **Trains a convolutional autoencoder** to reconstruct the clean image from the noisy input
4. **Visualizes** noisy vs. denoised vs. original images side by side

---

## Model Architecture

The Autoencoder follows a **symmetric Encoder-Decoder** structure with regularization for stability and generalization.

| Component | Layers | Description |
|-----------|--------|-------------|
| **Encoder** | `Conv2D`, `MaxPooling2D`, `Dropout`, `BatchNormalization` | Compresses the 28x28x1 grayscale image into a 7x7x16 latent space |
| **Decoder** | `Conv2D`, `UpSampling2D`, `Dropout`, `BatchNormalization` | Reconstructs the clean image from the compressed latent representation |
| **Output** | `Conv2D(1, sigmoid)` | Produces a single-channel image with pixel values constrained to [0, 1] |

```
Encoder:  Input(28x28x1) -> Conv2D(32) -> BN -> Dropout -> MaxPool
                          -> Conv2D(16) -> BN -> Dropout -> MaxPool -> Latent(7x7x16)

Decoder:  Conv2D(16) -> BN -> Dropout -> UpSampling
       -> Conv2D(32) -> BN -> Dropout -> UpSampling
       -> Conv2D(1, sigmoid) -> Output(28x28x1)
```

---

## Training Strategy

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss Function** | Mean Squared Error (MSE) |
| **Batch Size** | 128 |
| **Max Epochs** | 50 |
| **Early Stopping** | Patience = 5 (restores best weights) |
| **Noise Factor** | 0.5 (Gaussian) |
| **Data Augmentation** | Rotation, shifting, zooming via `ImageDataGenerator` |

---

## Design Choices and Why

### Why a Convolutional Autoencoder?

Autoencoders learn a compressed representation of the input and reconstruct it, naturally filtering out noise (which is hard to encode in a bottleneck). Using **convolutional layers** instead of dense layers preserves spatial relationships — critical for images — and dramatically reduces parameter count compared to a fully connected architecture.

### Why Fashion-MNIST?

Fashion-MNIST is a drop-in replacement for MNIST with more complex visual patterns (textures, edges, shapes of clothing). It provides a more meaningful benchmark for denoising than simple handwritten digits, while still being lightweight enough to train quickly without a GPU cluster.

### Why Gaussian Noise with Factor 0.5?

A noise factor of 0.5 is deliberately aggressive — it heavily corrupts the images, making the denoising task non-trivial. This forces the autoencoder to learn robust features rather than relying on a near-clean input. Values are clipped to [0, 1] after corruption to stay within valid pixel range.

### Why Batch Normalization?

BatchNorm stabilizes and accelerates training by normalizing intermediate activations. In autoencoders, where gradients must flow through both the encoder and decoder, this is especially important to prevent vanishing/exploding gradients and allows the use of higher learning rates.

### Why Dropout (0.25)?

Dropout acts as a regularizer, preventing the autoencoder from simply learning an identity mapping. By randomly zeroing 25% of activations during training, the network is forced to learn redundant, distributed representations — improving generalization to unseen noisy images.

### Why Data Augmentation?

`ImageDataGenerator` applies small random rotations, shifts, and zooms during training. This teaches the model to denoise images regardless of slight geometric variations, improving robustness. `horizontal_flip` is disabled because some Fashion-MNIST classes (like shoes) are not horizontally symmetric.

### Why Early Stopping?

Training runs for up to 50 epochs but stops early (patience = 5) if validation loss plateaus. This prevents overfitting — the model restores the weights from the best epoch, ensuring optimal generalization without manual epoch tuning.

### Why MSE Loss?

Mean Squared Error penalizes large per-pixel reconstruction errors, which directly aligns with the goal: minimize the difference between the denoised output and the clean original. For pixel-level reconstruction tasks, MSE is a natural and effective choice over alternatives like MAE (which treats all errors equally) or perceptual losses (overkill for 28x28 grayscale).

### Why Adam Optimizer?

Adam combines momentum and adaptive learning rates, converging faster and more reliably than vanilla SGD on this type of reconstruction task. It requires minimal hyperparameter tuning, making it a solid default for autoencoder training.

---

## Results

| Metric | Value |
|--------|-------|
| **Loss Function** | MSE |
| **Final Test Loss** | ~0.015 |

The low test loss indicates strong pixel-level reconstruction fidelity on unseen data.

### Image Reconstruction Grid (3x10)

| Row | Content |
|-----|---------|
| **1** | Noisy Input Images |
| **2** | Denoised Output (Model Predictions) |
| **3** | Original Clean Images |

![Uploading image.png…]()


---

## How to Run

### Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- TensorFlow 2.x

```bash
pip install numpy pandas matplotlib tensorflow
```

### Dataset

The project expects **Fashion-MNIST in CSV format** in the working directory:
- `fashion-mnist_train.csv`
- `fashion-mnist_test.csv`

Extract them from the included `archive.zip`, or download from [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist).

### Run

```bash
python "image denoising.py"
```

The script will:
1. Train the autoencoder (with early stopping)
2. Plot the training/validation loss curve
3. Display a comparison grid: **noisy -> denoised -> original** for 10 test images

---

## Project Structure

```
.
├── image denoising.py       # Full pipeline: data loading, model, training, visualization
├── archive.zip              # Fashion-MNIST dataset (CSV format)
└── README.md
```
