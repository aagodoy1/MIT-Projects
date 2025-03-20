import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""
#h = np.matrix('1. 1. 1.')
#h = np.matrix('1. 1.; 1. 1.; 1. 1.')
#print(f'{h.shape}')


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return max(x,0)
    # TODO

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    return 1 if x > 0 else 0
    # TODO

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS (Initialized to floats instead of ints)
        self.input_to_hidden_weights = np.matrix('1. 1.; 1. 1.; 1. 1.') #(3x2)
        self.hidden_to_output_weights = np.matrix('1. 1. 1.') # (1,3)
        self.biases = np.matrix('0.; 0.; 0.')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]

    def train(self, x1, x2, y):

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        #Input is input * weights + bias
        #(3x2) @ (2x1) + (3x1) = (3x1)
        hidden_layer_weighted_input = self.input_to_hidden_weights @ input_values + self.biases # TODO (3 by 1 matrix)
        hidden_layer_activation = rectified_linear_unit(hidden_layer_weighted_input)# TODO (3 by 1 matrix)

        #(1x3) @ (3x1)= (1x1)
        output =  np.dot(self.hidden_to_output_weights, hidden_layer_activation) # TODO #(Se multiplica un 1x3 por un 3x1)
        activated_output = output_layer_activation(output) # TODO

        ### Backpropagation ###

        # Compute gradients
        #Se usa la derivada de funcion de costo C = 1/2*(y-a)^2
        output_layer_error = activated_output - y # TODO
        hidden_layer_error = np.multiply(self.hidden_to_output_weights.T * output_layer_error, 
                                         rectified_linear_unit_derivative(hidden_layer_weighted_input)) # TODO (3 by 1 matrix)

# La fórmula general en backpropagation es:

# δ^l =(W^(l+1))^T * δ^(l+1)⊙f'(z^l)

# δ^l  es el error en la capa oculta (lo que queremos calcular).
# W^(l+1) es la matriz de pesos de la capa oculta a la salida (hidden_to_output_weights).
# 𝛿𝑙+1 es el error de la capa de salida (output_layer_error).
# f'(z^l) es la derivada de la función de activación en la capa oculta (usamos ReLU), evaluada en los valores que entraron a la capa oculta.
# 𝑧^l  es la entrada antes de aplicar la activación.

# Donde z^l se calcula como: 
# z^l=W^l * a^(l−1) + b^l

# W^l → pesos de la capa actual.
# a^(l−1) → activaciones de la capa anterior.
# 𝑏^l  → bias de la capa actual.
# Donde en este caso a^(l) = f(z^l)

        #El error o sesgo de la capa oculta, es el mismo que el de la neurona (3x1)
        # dC/db^l = δ^l
        bias_gradients = hidden_layer_error @  # TODO
        
        #dC/dW^l = δ^l * a^(l-1)
        hidden_to_output_weight_gradients = bias_gradients *  # TODO
        input_to_hidden_weight_gradients = # TODO

        # Use gradients to adjust weights and biases using gradient descent
        self.biases = # TODO
        self.input_to_hidden_weights = # TODO
        self.hidden_to_output_weights = # TODO

    def predict(self, x1, x2):

        input_values = np.matrix([[x1],[x2]])

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = # TODO
        hidden_layer_activation = # TODO
        output = # TODO
        activated_output = # TODO

        return activated_output.item()

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return


x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
# x.test_neural_network()
