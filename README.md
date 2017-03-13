#Still under construction

a few words for now, as we just recently moved a couple of days ago from Gitlab:

background and info:
https://docs.google.com/document/d/1ONzDBbF_XzeLUWmEfqhhtGX54NulEClrApoE9stfGsM/edit?usp=sharing
https://docs.google.com/presentation/d/1JoUyCv2KZjGTsuIV3v-UMdnsQctyD3cmlRtzQIXfSpQ/edit#slide=id.p4
----
Getting the code to work:

download and unzip any of the datasets mentioned

keras-posenet has a regular posenet implementation, inside scripts/posenet.py

lstm-keras-tf has a posenet+LSTM implementation, inside scripts/cnn_lstm.py

use the files in /scripts/exampleSettingsFile.py to set up your settings.py files required to run train.py and test.py

Our implementation steps:

We examined the following repository for a GoogleNet inception v1. Implementation using keras + Theano:

https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14

Which we based our Keras+TensorFlow implementation on

We changed the network architecture as described above

Preprocessed the frames of a scene by subtracting the mean image for each scene training set from all of the frames used for training

The same mean image is subtracted from the test set frames when testing

Each frame is cropped to the dimensions of 224x224, maintaining its aspect ratio

The cropping is carried out from the middle of the frame to ensure the landmark is present in the frame

The preprocessing steps implementation is present under utilities.py files in both of the directories

lstm-keras-tf and keras-posenet
