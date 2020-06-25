import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(X_train, _), (X_test, _) = mnist.load_data()  # loading dataset

X_test, X_train = X_test/255.0, X_train/255.0  # normalizing data

autoencoder = keras.models.load_model("autoencoder_snr=inf")  # loading the model from memory


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


show_reconstructions(autoencoder, 15)
plt.show()
