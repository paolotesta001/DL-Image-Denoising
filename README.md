# 👕 Fashion MNIST Denoising Autoencoder

An application of **Deep Learning** to clean and restore noisy images using a Keras/TensorFlow Autoencoder.

---

## 📖 Project Overview

This repository features a **Denoising Convolutional Autoencoder (CAE)** built with **TensorFlow** and **Keras**. The model's core function is to learn the inverse mapping of noise, 
effectively training it to remove **Gaussian noise** intentionally added to the popular **Fashion MNIST dataset**.

The result is a powerful model that leverages Deep Learning to perform **image restoration**, **noise reduction**, and **robust feature learning**, showcasing the practical use of Autoencoders 
for improving image quality and extracting stable features from corrupted data.

---

## ⚙️ Model Architecture

The Autoencoder follows a **symmetric Encoder–Decoder** structure with regularization for better stability and generalization.

| Component | Layers Used | Description |
|------------|--------------|--------------|
| **Encoder** | `Conv2D`, `MaxPooling2D`, `Dropout`, `BatchNormalization` | Compresses the 28×28×1 grayscale image into a 7×7×16 latent space. |
| **Decoder** | `Conv2D`, `UpSampling2D` | Reconstructs the clean image from the compressed latent representation. |
| **Regularization** | `Dropout`, `BatchNormalization` | Reduces overfitting and stabilizes training by mitigating internal covariate shift. |

### 🧠 Key Highlights
- Uses **ReLU** activations and **Sigmoid** output for normalized pixel intensity.  
- Employs **Adam optimizer** and **Mean Squared Error (MSE)** loss.  
- Includes **Early Stopping** to automatically restore the best-performing model.

---

## 🧩 Training and Data Strategy

1. **Noise Injection**  
   Gaussian noise (σ = 0.5) is added to simulate real-world image degradation.  

2. **Data Augmentation**  
   The model uses `ImageDataGenerator` to apply real-time transformations:
   - Rotation, shifting, and zooming for improved robustness.

3. **Training Configuration**  
   - **Batch Size:** 128  
   - **Epochs:** Up to 50  
   - **Early Stopping:** Patience = 5 (monitors validation loss)  

4. **Loss Function**  
   - **Mean Squared Error (MSE)** — minimizes pixel-wise differences between reconstructed and target images.

---

## 📊 Results & Evaluation

The model achieves **excellent denoising performance**, producing clean, visually faithful outputs with minimal reconstruction error.

| Metric | Value | Description |
|---------|--------|-------------|
| **Loss Function** | MSE | Measures pixel-wise reconstruction fidelity. |
| **Final Test Loss** | ~0.015 | Indicates strong denoising and high accuracy on unseen data. |

---

## 🖼️ Visualizations

**1️⃣ Loss Trend**  
Displays *Training vs. Validation Loss* per epoch to show convergence and early stopping.  

**2️⃣ Image Reconstruction Grid (3×10)**  

| Row | Description |
|------|--------------|
| **1** | Noisy Input Images |
| **2** | Denoised Output (Model Predictions) |
| **3** | Original Clean Images |

📸 *Example visualization of the denoising effect:*




