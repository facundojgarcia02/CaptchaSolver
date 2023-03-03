import tensorflow as tf
import os

from random import shuffle
from utils.preprocess import encode_single_sample, get_filename

def create_dataset(filenames: list[str], 
                   labels: list[str], 
                   batch: int = 4) -> tf.data.Dataset:
    '''
    Crear dataset y aplicar transformaciones a partir de la lista de imagenes y los labels.
    '''

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(buffer_size = tf.data.AUTOTUNE)

    return dataset

def load_dataset(folder: str = 'samples', test_size: float = 0.2):
    '''
    Cargar nombres de imagenes. Devuelve los nombres y los labels transformados.
    '''

    _, _, images = list(os.walk(folder))[0]
    # Agregar path completo a los archivos.
    images = list(map(lambda x: folder + x, images))
    shuffle(images)

    samples = len(images)
    train_samples = int(samples*test_size)

    # Split de train y validación.
    train_images = images[:train_samples]
    validation_images = images[train_samples:]

    # Aplicar preprocesamiento para conseguir etiquetas.
    train_labels = list(map(get_filename, train_images))
    validation_labels = list(map(get_filename, validation_images))

    return train_images, validation_images, train_labels, validation_labels