import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import datetime
# open tensorboard with "tensorboard --logdir logs/fit" in the Folder where the logs Folder is
import numpy as np
# turn off eager execution
# tf.compat.v1.disable_eager_execution()
# ----------------------------------------Tensorboard------------------------------------------------
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "with_complex_encoding"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# ---------------------------------------------------------------------------------------------------
# print(tf.test.gpu_device_name())
n_reduced = 1024  # how many bits will be transmitted bei 512 komprimierung um Faktor 12.25

(X_train, _), (X_test, _) = mnist.load_data()  # loading the mnist dataset

X_test, X_train = X_test/255.0, X_train/255.0  # normalizing the data


stacked_encoder = keras.models.Sequential([  # building the encoder

    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(200, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(n_reduced, activation="sigmoid"),
    keras.layers.Dropout(0.3)
])

stacked_decoder = keras.models.Sequential([  # building the decoder
    keras.layers.Dense(100, activation="selu", input_shape=[n_reduced]),
    keras.layers.Dense(200, activation="selu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])


class Emitter(tf.keras.layers.Layer):
    def __init__(self, num_outputs, units=32, input_dim=32):  # TODO: add parameter that switches between continuous and discrete
        super(Emitter, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        pass

    def call(self, input):
        # return input
        # return tf.math.ceil(input)
        # clipped_input = tf.clip_by_value(input, clip_value_min=-100, clip_value_max=100)
        rounded = tf.math.round(input)  # TODO: map each input float to 2 integers, that bet converted to complex numbers
        reshaped_input = tf.reshape(rounded, [2, -1])
        # print(tf.shape(reshaped_input))
        real = reshaped_input[0]
        imag = reshaped_input[1]
        return tf.complex(real, imag)


class Noise(tf.keras.layers.Layer):
    def __init__(self, num_outputs, units=32, input_dim=32):
        super(Noise, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        pass

    def call(self, input):
        real = tf.math.real(input)  #TODO: add noise. Use GaussianNoise on 0 vector of the same shape
        imag = tf.math.imag(input)
        real_noise = keras.layers.GaussianNoise(0.5)(real, training=True)
        imag_noise = keras.layers.GaussianNoise(0.5)(imag, training=True)
        return tf.complex(real_noise, imag_noise)


class Receiver(tf.keras.layers.Layer):
    def __init__(self, num_outputs, ):
        super(Receiver, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        pass

    def call(self, input):
        real = tf.math.real(input)
        imag = tf.math.imag(input)
        concat = tf.concat([real, imag], axis=0, name='concatenate')
        return tf.round(concat)


# testing the instantiation of layers
emitter = Emitter(30)
noise = Noise(num_outputs=30, units=30, input_dim=30)
receiver = Receiver(30)

# testing the call method
print("Testing Emitter:")
print("Output should be: [1.+4.j 2.+5.j 3.+6.j]")
t1 = [[1., 2., 3., 4., 5., 6.]]
emitter_output = emitter(t1)
print(emitter_output)  # should return [1.+4.j 2.+5.j 3.+6.j]

# output = emitter(tf.zeros([40]))  # Calling the layer `.builds` it.
# print(output)
# print(noise(output))
# output = gauss_noise = keras.layers.GaussianNoise(0.5)(output, training=True)
print("Testing noise Layer:")
noise_output = noise(emitter_output)
print(noise_output)
print("Testing receiver:")

receiver_output = receiver(noise_output)  # Calling the layer `.builds` it.
print(receiver_output)


# building a sequential model
channel = keras.models.Sequential([
    keras.layers.InputLayer(n_reduced),
    Emitter(n_reduced),
 #   Noise(n_reduced),
    Receiver(n_reduced)])

# channel.build(input_shape=[n_reduced])
channel.summary()

output = channel.predict(tf.zeros([30]))

print(output)
# ------------------------------------------Pretraining--------------------------------------------------
# Gaussian noise to make the model choose more extreme values close to 0 and 1
pretraining_model = keras.models.Sequential([stacked_encoder, keras.layers.GaussianNoise(0.7), stacked_decoder])
pretraining_model.summary()
pretraining_model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.5))
print("Starting pretraining:")
pretraining_model.fit(X_train, X_train, epochs=10, validation_data=(X_test, X_test), callbacks=[tensorboard_callback])
# ------------------------------------------simulation---------------------------------------------------
simulator = keras.Sequential([stacked_encoder, channel, stacked_decoder])
simulator.summary()
simulator.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.5))

history = simulator.fit(X_train, X_train, epochs=5, validation_data=(X_test, X_test))

simulator.save('autoencoder')  # saving autoencoder so we don't have to train it every time


def plot_image(image):
    plt.imshow(image)
    plt.axis("off")


def show_reconstructions(model, n_images=8):
    reconstructions = model.predict(X_test[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(X_test[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])


show_reconstructions(simulator, n_images=15)
plt.show()
