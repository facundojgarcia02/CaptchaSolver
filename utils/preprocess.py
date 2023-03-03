import os
import tensorflow as tf
from keras.layers import StringLookup

from utils.constants import DICTIONARY, IMAGE_HEIGHT, IMAGE_WIDTH

# Convertir caracteres a números
char_to_num = StringLookup(
    vocabulary=list(DICTIONARY.keys()), mask_token=None
)

num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def get_filename(path):
    '''
    Conseguir nombre de la imagen desde el path de la misma.
    '''

    filename = path.split("/")[-1]
    filename_wo_ext = filename[:-4]

    return filename_wo_ext

def encode_single_sample(img_path, label):
    '''
    Transformación del label y carga de la imagen.
    '''
    
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])

    # Transponer la imagen para que el ancho sea el eje temporal
    # (Para el orden en los caracteres)
    img = tf.transpose(img, perm=[1, 0, 2])

    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

    return {"image": img, "label": label}

