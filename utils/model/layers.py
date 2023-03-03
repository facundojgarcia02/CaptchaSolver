import tensorflow as tf
from tensorflow import keras

class CTCLayer(keras.layers.Layer):
    '''
    Layer personalizada para calcular la pérdida CTC.

    Referencia:
    https://keras.io/examples/vision/captcha_ocr/#model
    '''

    def __init__(self, name=None, **kwargs):
        super(CTCLayer, self).__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):

        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        # Agregar la pérdida al modelo utilizando Layer.add_loss(loss)
        self.add_loss(loss)

        return y_pred