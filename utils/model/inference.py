import tensorflow as tf
import numpy as np

from tensorflow import keras
from utils.preprocess import num_to_char
from utils.model.layers import CTCLayer

def decode_batch_predictions(pred):
    '''
    Decodificar output de la red.
    '''
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :5]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

class InferenceModel():
    '''
    Modelo para inferencia de Captchas, creado utilizando
    el modelo ya entrenado.
    '''
    def __init__(self, trained_model):
        
        # Si se pasa el filepath, cargar el model.
        if isinstance(trained_model, str):
            trained_model = keras.models.load_model(trained_model, custom_objects = {'CTCLayer': CTCLayer()})

        self.model = keras.models.Model(trained_model.get_layer('image').input, 
                                    trained_model.get_layer('dense2').output)
    
    def predict(self, images):
        preds = self.model.predict(images)
        pred_texts = decode_batch_predictions(preds)

        return pred_texts

    def __call__(self, images):
        preds = self.model(images)
        pred_texts = decode_batch_predictions(preds)

        return pred_texts