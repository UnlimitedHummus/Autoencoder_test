import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist

print(tf.test.gpu_device_name())

(X_train, _), (X_test, _) = mnist.load_data()

# normalize
X_test, X_train = X_test/255.0, X_train/255.0


stacked_encoder = keras.models.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu")
])

stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

stacked_autoencoder = keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_autoencoder.summary()
stacked_autoencoder.compile(loss="binary_crossentropy",
                            optimizer=keras.optimizers.SGD(lr=1.5))

history = stacked_autoencoder.fit(X_train, X_train, epochs=10,
                                  validation_data=(X_test, X_test))

stacked_autoencoder.save('autoencoder')  # saving autoencoder so we don't have to train it every time
