# Dog Breed Classifier

This project uses a convolutional neural network to detect and
classify dogs into 133 different breeds.  The use of Haar cascades is
demonstrated. There is also a mutt-detector section, where if the dog
seen in the photo is classified to a single breed with less than 90%
confidence, the network outputs the other breeds that it thinks form
part of the ancestry of the dog. As a lighter aside, there is a
section to accept photos of human beings (celebrities in this
notebook) and the network, after deciding that it is not a dog, gives
an estimate of which categogy of dog the person most closely
resembles.

Performance on three neural networks is demonstrated - one is
ResNet-50, deep residual neural network, pretrained on ImageNet
specifically for this task, another is a CNN built from scratch, and
finally, uses transfer learning from.

## The dataset.

The training set consists of around 6,700 images of dogs in 133
categories.  The validation and test sets are about 835 images each. 

## Using the Resnet50 model trained on ImageNet

The residual network
[Resnet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)
is used here with weights which have been trained on the 10-million
image set [ImageNet](http://www.image-net.org/) to output one of 1000
classes of objects, including types of dogs.  As a dog detector,
simply noting if the image belongs to one of the 133 types of dogs or
not, but ignoring which class the output belongs to, on a test set of
100 human faces and 100 dog faces, this network had perfect
performance.

## Convolutional network from scratch

The network architecture consist of three convolutional layers, each
followed by max-pooling and incorporating dropout, as well as a fourth
convolutional layer that uses global average pooling.  This final
flattened layer is connected to a fully connected (dense) layer of 133
nodes and a softmax activation. The accuracy of this network is over 81%. 

## Transfer learning

Here, a model trained on other images is connected to a GAP
layer (global average pooling) and then to a fully connected layer
with softmax activation.  During training, only the added layers are trained, *i.e.*, the pretrained weights are frozen.  

## Losses and optimizer and used of the best model

* The training is with categorical cross entropy, which is suitable
for a classification problem like this.

* I have used the RMSprop optimizer although Adam would also be a good
choice.

* The performance is tested on the validation set after each epoch of
training weights are stored if they are found to improve on the best
weights seen until that point.

## Acknowledgements

These networks are implemented using **Keras** with TensorFlow as
backend. The project was done as work towards the Udacity nanodegree
program and Udacity's original license is included here.