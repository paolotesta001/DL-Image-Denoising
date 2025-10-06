# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 22:03:21 2025

@author: Utente
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Carica i dati di training e test da file CSV locali
train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

#Estrae le immagini e le normalizza tra 0 e 1
x_train = train_df.iloc[:, 1:].values.astype("float32") / 255.
x_test = test_df.iloc[:, 1:].values.astype("float32") / 255.

#Reshape in formato immagine (N, 28, 28, 1) per usare convoluzioni
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#Aggiunta del rumore
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#Data Augmentation
#Crea un ImageDataGenerator per augmentation di immagini rumorose e originali
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False
)

# Generatore per immagini rumorose e pulite
train_gen = datagen.flow(x_train_noisy, x_train, batch_size=128)

#Costruzione dell'Autoencoder con Dropout e BatchNormalization

input_img = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)
x = layers.UpSampling2D((2, 2))(x)

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)
x = layers.UpSampling2D((2, 2))(x)

decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

#Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


#Training del modello con Data Augmentation

history = autoencoder.fit(
    train_gen,
    steps_per_epoch=len(x_train) // 128,
    epochs=50,
    validation_data=(x_test_noisy, x_test),
    callbacks=[early_stopping]
)

# Grafico early stopping
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss (Early Stopping)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Inferenza e visualizzazione

decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(18, 7))
for i in range(n):
    #Riga 1: immagini rumorose
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    if i == 0:
        ax.set_ylabel("Noisy")
    #Riga 2: immagini denoised (output modello)
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    if i == 0:
        ax.set_ylabel("Denoised")
    #Riga 3: immagini originali
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    if i == 0:
        ax.set_ylabel("Original")
plt.tight_layout()
plt.show()