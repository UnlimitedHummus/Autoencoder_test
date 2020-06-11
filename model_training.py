import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist

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

stacked_autoencoder = keras.models.Sequential([stacked_encoder, stacked_decoder])  # putting autoencoder together
stacked_autoencoder.summary()  # printing summary of model
stacked_autoencoder.compile(loss="binary_crossentropy",
                            optimizer=keras.optimizers.SGD(lr=1.5, momentum=0.9))  # compiling model

history = stacked_autoencoder.fit(X_train, X_train, epochs=30,
                                  validation_data=(X_test, X_test))  # training model

stacked_autoencoder.save('autoencoder')  # saving autoencoder so we don't have to train it every time
