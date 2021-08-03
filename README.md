# Autoencoder trained for transmission over AWGN channel
This repository houses an Autoencoder built using Tensorflow. It's supposed to optimize transmission over a lossy channel. The Autoencoder is Trained on the MNIST Dataset and the project should serve as a proof of concept. 

The Simulation folder houses `training.py`, `testing.py` a folder for tensorboard logs and a folder where the model is saved.
In `training.py` the model gets trained
In `testing.py` a few predictions get made with the model and you can look at how the transmitted images compare to the original ones. 
