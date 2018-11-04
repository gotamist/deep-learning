# Bike Sharing

This project is an example of a multi-level perceptron (vanilla
neural network) for a regression problem.

## Data

The dataset has the number of riders using a bike sharing company over
a two year period. The daily data as well as the hourly data is
available.  There are 59 variables, including several season dummies.

## The network

The neural network is implemented from scratch (*i.e.*, the
optimization is not performed using any programming framework like
TensorFlow or PyTorch), but by stochastic gradient descent by
implementing the backpropagation here. The network class is shown in
the file `my_answers.py`.

This network has a single hidden layer.  Hyperparameters are chosen by
trying various values and observing the decay of the optimizer
training and validation loss wih the number of iterations.

## Results

There is a comparison of the of the predicted bike-use with
the actual data and an investigation into why there is difficulty
matching the data during the holiday season.

This project was done as work towards the Udacity Deep Learning Nanodegree.

