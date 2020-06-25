import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import datetime

# ----------------------------------------Tensorboard------------------------------------------------
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# ---------------------------------------------------------------------------------------------------

n_reduced = 1024  # how many bits will be transmitted choose a multiple of 2
# bei 512 komprimierung um Faktor 12.25

(X_train, _), (X_test, _) = mnist.load_data()  # loading the mnist dataset

X_test, X_train = X_test/255.0, X_train/255.0  # normalizing the data

# ------------------------------------------Autoencoder-----------------------------------------------

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
# ------------------------------------------ChannelLayer-------------------------------------------------


class ChannelLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, snr=np.inf):  #TODO: add SNR and disables/ enables rounding
        super(ChannelLayer, self).__init__()
        self.num_outputs = num_outputs
        self.snr = snr

    def build(self, input_shape):
        pass

    def call(self, input):
        # batch_size = tf.shape(input)[0]
        # rounding makes the input take discrete values 0 or 1
        rounded_0 = tf.round(input)  # TODO: add parameter that enables and disables rounding

        # Make first half real part, second half imaginary part
        real_0, imag_0 = tf.split(rounded_0, num_or_size_splits=2, axis=1)
        complex_0 = tf.complex(real_0, imag_0)
        # simulate sending data over noisy channel
        # TODO: add SNR ----------------------------------------------------------------------------
        signal_power = tf.math.real((tf.norm(complex_0)**2)) / tf.cast(n_reduced//2, tf.dtypes.float32)
        noise_power = signal_power / self.snr
        halved = noise_power / 2.0
        scaling = tf.math.sqrt(halved)
        noise = tf.complex(scaling, [0.]) * tf.complex((tf.random.normal(shape=[n_reduced//2])), tf.random.normal(shape=[n_reduced//2]))
        complex_2 = complex_0 + noise

        # -------------------------------------------------------------------------------------------
        real_noise = keras.layers.GaussianNoise(0.7)(tf.fill(tf.shape(real_0), 0.0), training=True)
        imag_noise = keras.layers.GaussianNoise(0.7)(tf.fill(tf.shape(imag_0), 0.0), training=True)
        complex_noise = tf.complex(real_noise, imag_noise)
        complex_1 = complex_0 + complex_noise  # adding complex noise

        # turing the complex data back into real data
        real_1 = tf.math.real(complex_2)
        imag_1 = tf.math.imag(complex_2)

        # concatenate back to the original shape
        concat_data = tf.concat([real_1, imag_1], axis=1)

        # round to get back to discrete values
        rounded_1 = tf.round(concat_data)
        output = rounded_1
        if False:
            print("shape before sending:", rounded_0)
            print("Real_0 part: ", real_0)
            print("Real_1 part:", real_1)
            print("Shape after transmitting", concat_data)
        return output


# ------------------------------------------Pretraining--------------------------------------------------
# Gaussian noise to make the model choose more extreme values close to 0 and 1
def pretraining(epochs=2):
    pretraining_model = keras.models.Sequential([stacked_encoder, keras.layers.GaussianNoise(0.7), stacked_decoder])
    pretraining_model.summary()
    pretraining_model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.5))
    print("Starting pretraining:")
    pretraining_model.fit(X_train, X_train, epochs=epochs, validation_data=(X_test, X_test), callbacks=[tensorboard_callback])


# pretraining(30)
# ------------------------------------------simulation---------------------------------------------------
simulator = keras.Sequential([stacked_encoder, ChannelLayer(n_reduced, snr=np.inf), stacked_decoder])
simulator.summary()
simulator.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.5))
history = simulator.fit(X_train, X_train, epochs=2, validation_data=(X_test, X_test))
simulator.save('autoencoder_snr=inf')  # saving autoencoder_snr=inf so we don't have to train it every time

# -----------------------------------------show images---------------------------------------------------


def plot_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis("off")


def show_reconstructions(model, n_images=5):  # showing images before and after autoencoding
    reconstructions = model.predict(X_test[:n_images])  # sending images through autoencoder_snr=inf
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):  # plotting images vs autoencoded images
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(X_test[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])


show_reconstructions(simulator, 8)
plt.show()
