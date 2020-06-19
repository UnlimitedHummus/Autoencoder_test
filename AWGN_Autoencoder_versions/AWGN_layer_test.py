import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
# from Transmitters.qpsk import QPSKTransmitter
from komm import AWGNChannel
from math import pi as const_pi

# print(tf.test.gpu_device_name())
n_reduced = 30

(X_train, _), (X_test, _) = mnist.load_data()  # loading the mnist dataset

X_test, X_train = X_test / 255.0, X_train / 255.0  # normalizing the data

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
# transmitter = QPSKTransmitter(1)
channel = AWGNChannel(snr=1.0)


class AWGNLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(AWGNLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        # VARIANTE 1

        # Make first half real part, second half imaginary part
        data = tf.dtypes.complex(inputs[:n_reduced // 2], inputs[n_reduced // 2 + 1:])
        # Transmit over AWGN channel
        data = channel(data)
        # extract real part
        real_part = tf.dtypes.cast(data, tf.float32)
        # extract imaginary part by flipping real and imaginary parts
        temp_data = tf.math.multiply(data, 0 - 1j)
        imag_part = tf.dtypes.cast(temp_data, tf.float32)
        # Concatenate back to original shape
        data = tf.concat([real_part, imag_part], 1)
        return data

    def call2(self, inputs, **kwargs):
        # VARIANTE 2

        # Make first half real part, second half imaginary part
        angles = inputs[:n_reduced // 2]
        amplitudes = inputs[n_reduced // 2 + 1:]
        # Confine angles to [0, 2*pi) and amplitudes to (0,1)
        angles = tf.math.floormod(angles, 2 * const_pi)
        amplitudes = tf.math.sigmoid(amplitudes)
        # Transmit over AWGN channel
        angles = channel(angles)
        amplitudes = channel(amplitudes)
        # Convert polar coordinates back to floats
        # ...
        return


layer = AWGNLayer(15)
output = layer(tf.zeros([40]))  # Calling the layer `.builds` it.
print(output)

# ----------------------------------------------------------------------------------------------------------


stacked_autoencoder = keras.models.Sequential([stacked_encoder, stacked_decoder])  # putting autoencoder together
stacked_autoencoder.summary()  # printing summary of model
stacked_autoencoder.compile(loss="binary_crossentropy",
                            optimizer=keras.optimizers.SGD(lr=1.5, momentum=0.9))  # compiling model

history = stacked_autoencoder.fit(X_train, X_train, epochs=30,
                                  validation_data=(X_test, X_test))  # training model

stacked_autoencoder.save('autoencoder_without_white_noise')  # saving autoencoder

channel_encoder = keras.models.Sequential([stacked_encoder, AWGNLayer, stacked_decoder])
