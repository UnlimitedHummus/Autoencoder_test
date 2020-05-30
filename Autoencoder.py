import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import sklearn
from sklearn.datasets import load_digits

print("Tensorflow version", tf.__version__)

print(tf.test.gpu_device_name())

(mnist_train, mnist_test), mnist_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,  # changed from True
    with_info=True,
)
mnist_test = tfds.as_numpy(mnist_test)
print(type(mnist_test))

exit(0)


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


mnist_train = mnist_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
mnist_train.cache()
mnist_train.shuffle(mnist_info.splits['train'].num_examples)
mnist_train = mnist_train.batch(128)
mnist_train = mnist_train.prefetch(tf.data.experimental.AUTOTUNE)


mnist_test = mnist_test.map(
    normalize_img, num_parallel_calls= tf.data.experimental.AUTOTUNE)
mnist_test = mnist_test.batch(128)
mnist_test = mnist_test.cache()
mnist_test = mnist_test.prefetch(tf.data.experimental.AUTOTUNE)

stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu")
])

stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

stacked_autoencoder = keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_autoencoder.compile(loss="binary_crossentropy",
                            optimizer=keras.optimizers.SGD(lr=1.5))

history = stacked_autoencoder.fit(mnist_train, mnist_train, epochs=10,
                                  validation_data=[mnist_test, mnist_test])
