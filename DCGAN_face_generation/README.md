# Generating human faces with Deep Convolutional Generative  Adversarial Networks.

## Introduction
GANs are one of the most exciting innovations in machine learning in
recent years.  They were first introduced by [Goodfellow et
al](https://arxiv.org/abs/1406.2661) in 2014.  In the task of learning
probability distributions, much of the success prior to that came in
the form of discriminative models, of the kind used in most of the
problems in these repositories.  Success had been much more limited
with generative models.  The idea was to simultaneously train two
models: a generative model G that captures the data distribution, and
a discriminative model D that estimates the probability that a sample
came from the training data rather than G. The training procedure for
G is to maximize the probability of D making a mistake.  

While the original work was implemented using MLPs (vanilla neural
nets), further work by [Radford et
al](https://arxiv.org/pdf/1511.06434.pdf) in 2016 introduced a class
of CNNs called DCGANs.  Here, both D and G are CNNs. The work in this
directory is an implementation of such an architecture.

## Distributions and datasets

Images are generated from two distributions:

1. Handwritten digits from the [MNIST Database of handwritten
digits](http://yann.lecun.com/exdb/mnist/).
2. [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), a
large-scale celebrity faces attributes dataset.

## Files

The main notebook in this directory is `digit_and_face_generation.ipynb`. Some helper functions for getting and
preprocessing data are in `helpers.py`. Unit tests provided by Udacity
are in `problem_unittests.py`.

The file `smaller_lr_beta_digit_face_generation.ipynb` is included to demonstrate the
performance that can already be observed with poorer training (smaller
learning rate and momentum beta).

## Architecture

The programming framework used is Tensorflow. Here GAN is contructed
with a D that uses 3 convolutional layers and a G that uses 4 of them.
Both the D and G use batch normalization, a ReLU (Rectified Linear
Unit) activation and dropout layers for regularization on each
convolutional layer. The logits of the generator are passed through a
`tanh` activation function while the discriminator uses a sigmoid
activation.

## Acknowledgements

This project was done as part of the Deep Learning Nanodegree at
Udacity.  The provision of starter framework code is gratefully
acknowledged, which freed me to focus on the network architecture,
loss criterion definition, optimizer selection, regularization and
hyperparameter tuning.