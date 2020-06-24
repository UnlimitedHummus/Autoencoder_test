import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from Transmitters.qpsk import QPSKTransmitter
import numpy as np
# turn off eager execution
# tf.compat.v1.disable_eager_execution()

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


class Emitter(tf.keras.layers.Layer):
    def __init__(self, num_outputs, units=32, input_dim=32):
        super(Emitter, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        pass

    def call(self, input):
        return input


class Noise(tf.keras.layers.Layer):
    def __init__(self, num_outputs, units=32, input_dim=32):
        super(Noise, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        pass

    def call(self, input):
        return input


class Receiver(tf.keras.layers.Layer):
    def __init__(self, num_outputs, ):
        super(Receiver, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        pass

    def call(self, input):
        return input


# testing the instanciation of layers
emitter = Emitter(30)
noise = Noise(num_outputs=30, units=30, input_dim=30)
receiver = Receiver(30)

# testing the call method
output = emitter(tf.zeros([40]))  # Calling the layer `.builds` it.
print(output)
output = noise(tf.zeros([40]))  # Calling the layer `.builds` it.
print(output)
output = receiver(tf.zeros([40]))  # Calling the layer `.builds` it.
print(output)

# building a sequential model
channel = keras.models.Sequential([Emitter(30), Noise(30), Receiver(30)])
channel.build(input_shape=[30])
channel.summary()
