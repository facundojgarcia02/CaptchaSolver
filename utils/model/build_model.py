from keras import layers
from tensorflow import keras

from utils.constants import IMAGE_HEIGHT, IMAGE_WIDTH
from utils.preprocess import char_to_num

from utils.model.layers import CTCLayer

def build_model(learning_rate = 1e-3):
    '''
    Estructura tomada de referencia de:

    https://keras.io/examples/vision/captcha_ocr
    '''

    # Inputs
    input_img = layers.Input(
        shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # Bloque de convoluciones.
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Bloque de convoluciones
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Reshape de la ultima capa (Entre los strides y los 
    # MP2D dividimos por 4 el tamaño y tenemos 64 filtros 
    # en la última Conv2D)
    new_shape = ((IMAGE_WIDTH // 4), (IMAGE_HEIGHT // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Capa con la función de perdida CTC
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Armar modelo y compilar.
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    opt = keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(optimizer=opt)

    return model

if __name__ == "__main__":
    model = build_model(learning_rate= 0.001)
    print(model.summary())