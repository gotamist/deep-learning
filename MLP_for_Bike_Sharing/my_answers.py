import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: DONE: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1.0/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backpropagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: DONE Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function( hidden_inputs ) # signals from hidden layer

        # TODO: DONE Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)#signals into final output layer
        final_outputs =  final_inputs  # activation function for output node is the identity
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: DONE Output error - Replace this value with your calculations.
        #print( "shape of final_outputs",final_outputs.shape)
        error = y-final_outputs # Output layer error is the difference between desired target and actual output.
              
        # TODO: DONE Backpropagated error terms - Replace these values with your calculations.
        output_error_term =  error #1 is the derivative of the identity function. I am just making that explicit by introducing the 1 here
        
        # TODO: DONE Calculate the hidden layer's contribution to the error
        hidden_error = output_error_term * self.weights_hidden_to_output   
        activation_prime=hidden_outputs*(1-hidden_outputs)
        hidden_error_term = hidden_error*activation_prime[:,None]
        # Weight step (input to hidden)
        n_inputs = X.shape[0]
        x=X.reshape(1,n_inputs) #matrix math is just easier for me
        delta_weights_i_h += x.T*hidden_error_term.T  
        # Weight step (hidden to output)
        #print("hidden_outputs has shape",hidden_outputs.shape)
        n_outputs=final_outputs.shape[0]
        #print("number of final outputs = ",n_outputs)
        oet=output_error_term.reshape(1,n_outputs)
        #print("oet has shape",oet.shape)
        h_op=hidden_outputs.reshape(hidden_outputs.shape[0],1)
        delta_weights_h_o += h_op*oet #in terms of 2-D arrays
        #print( delta_weights_h_o.shape )
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: DONE Hidden layer - Replace these values with your calculations.

        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: DONE Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)#signals into final output layer
        final_outputs =  final_inputs  # recall that for the final layer, the output activation function is the identity
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################



iterations = 6000 
learning_rate = 0.3
hidden_nodes = 6
output_nodes = 1

# I found very similar results with the following hyperparameters

#iterations = 5000
#learning_rate = 0.4
#hidden_nodes = 4
#output_nodes = 1

# This was not the best in performance (on training or validation) 
#iterations = 5000 
#learning_rate = 0.05
#hidden_nodes = 2
#output_nodes = 1

# I have tried much smaller learning rates, such as 0.02, but I found that in this problem, there isn't much gain by keeping a low learning rate, which is how I first proceeded out of caution.  By having a higher learning rate, even though the loss-iterations graphs is not super-smooth, I feel that the code is able to go further along in the search process for the same amount of computational expense.  The training error comes down to about 0.1 and the validation error to about 0.2 in each of the cases above.  The performance on the test data is good on all days except december 22-26 where the model overpredicted.  I think that is because, unlike most of the training days, these days are around Christmas (no dummy variable captures that) when real demand would have been quite low.

# I could see that computational expense is much smaller when the hidden layer has only 2 nodes.  However, training error never really gets below 0.25 and validation error never under 0.4.  This suggests that the n_hidden=2 is too crude a model.  n_hidden =3 or 4 would be better.  Maybe just 3 would be preferable due to the lower complexity.  I did try 5 hidden nodes also, but there was no benefit and I would never choose that.

