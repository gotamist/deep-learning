# Dog Breed Classifier

This project uses a convolutional neural network to detect and
classify dogs into 133 different breeds.  As a lighter aside, there is
a section to accept photos of human beings (celebrities in this
notebook) and the network, after deciding that it is not a dog, gives
an estimate of which categogy of dog the person most closely
resembles.  The use of Haar cascades is demonstrated. There is also a
mutt-detector section, where if the dog seen in the photo is
classified to a single breed with less than 90% confidence, the
network outputs the other breeds that it thinks form part of the
ancestry of the dog.

Performance on two neural networks is demonstrated - once is built from
scratch and another uses transfer learning from a pretrained deep residual net.

## The dataset.

The training set consists of around 6,700 images of dogs in 133
categories.  The validation and test sets are about 835 images each. 

## Transfer learning

The residual network
[Resnet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)
which has been trained on the 10-million image set
[ImageNet](http://www.image-net.org/)