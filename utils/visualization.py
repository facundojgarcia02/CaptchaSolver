import matplotlib.pyplot as plt
import tensorflow as tf

from numpy import ndarray
from utils.preprocess import num_to_char

def plot_examples(dataset: tf.data.Dataset) -> ndarray:
    '''
    Mostrar ejemplos de imagenes del dataset.
    '''
    
    _, ax = plt.subplots(4, 4, figsize=(10, 5))
    for i, batch in enumerate(dataset.take(4)):
        images = batch['image']
        labels = batch['label']

        for j in range(4):
            img = (images[j] * 255).numpy().astype("uint8")
            label = tf.strings.reduce_join(num_to_char(labels[j])).numpy().decode("utf-8")
            ax[i, j].imshow(img[:, :, 0].T, cmap="gray")
            ax[i, j].set_title(label)
            ax[i, j].axis("off")

    plt.show()

    return ax

def plot_loss(history):
    '''
    Graficar curvas de la función de perdida.
    '''

    _, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.plot(history.history['loss'], label="Loss")
    ax.plot(history.history['val_loss'], label="Validation Loss")

    ax.legend(fancybox=False, loc="upper right", framealpha=1, edgecolor="black")
    ax.grid(ls="--", color="gray")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, len(history.history['loss']))

def plot_predictions(dataset: tf.data.Dataset, 
                     inference_model: tf.keras.Model) -> ndarray:
    '''
    Mostrar ejemplos de predicciones sobre un dataset dado con un modelo entrenado.
    '''

    _, ax = plt.subplots(4, 4, figsize=(10, 5))

    # Hacemos prediccione sobre 4 batches
    for i, batch in enumerate(dataset.take(4)):

        images = batch['image']
        preds = inference_model.predict(images)

        # Plotear cada imagen junto a su predicción en una fila.
        for j in range(4):
            img = (images[j] * 255).numpy().astype("uint8")
            ax[i, j].imshow(img[:, :, 0].T, cmap="gray")
            ax[i, j].set_title(f'Prediction = {preds[j]}')
            ax[i, j].axis("off")

    plt.show()

    return ax
