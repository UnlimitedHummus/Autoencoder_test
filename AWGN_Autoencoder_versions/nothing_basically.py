import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from Transmitters.qpsk import QPSKTransmitter
import numpy as np
# turn off eager execution
tf.compat.v1.disable_eager_execution()

# print(tf.test.gpu_device_name())
n_reduced = 30

(X_train, _), (X_test, _) = mnist.load_data()  # loading the mnist dataset

X_test, X_train = X_test/255.0, X_train/255.0  # normalizing the data


stacked_encoder = keras.models.Sequential([  # building the encoder

    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(n_reduced, activation="selu")
])

stacked_decoder = keras.models.Sequential([  # building the decoder
    keras.layers.Dense(100, activation="selu", input_shape=[n_reduced]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
# ----------------------------------------------------------------------------------------------------------
transmitter = QPSKTransmitter(1)


class AWGNLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(AWGNLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs])

    def call(self, input):
        # print(type(input.numpy()))
        input_array = input.eval()
        scaled_array = input_array*255
        int_array = scaled_array.astype(np.uint8)
        transmitted_array = transmitter.transmit_byte_array(int_array) / 255
        final_tensor = tf.convert_to_tensor(transmitted_array)
        return final_tensor


with tf.compat.v1.Session() as sess:
    layer = AWGNLayer(40)

    output = layer(tf.zeros([40]))  # Calling the layer `.builds` it.
    print(output)

    transmitter_layer = keras.models.Sequential([
        AWGNLayer(n_reduced)
    ])
    # channeled_encoder = keras.models.Sequential([stacked_encoder, transmitter_layer])
    # channeled_encoder = keras.models.Sequential(transmitter_layer, stacked_decoder)
    transmitter_layer.compile()
# ----------------------------------------------------------------------------------------------------------
