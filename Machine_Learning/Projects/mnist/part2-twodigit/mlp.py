import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
#sys.path.append(r"C:\Users\user\Documents\GitHub\MIT-Projects\Machine_Learning\Projects\mnist")
#path_to_data_dir = '../Datasets/'
path_to_data_dir = 'C:/Users/user/Documents/GitHub/MIT-Projects/Machine_Learning/Projects/mnist/Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions
'''
class MLP(nn.Module):

    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        #Se crea una sola capa oculta que es compartida por ambos modelos para que el vector W de pesos se entrene tomando ambos en cuenta
        # Esto lo hacemos solo porque los numeros vienen de una misma imagen, si vinieran de 2 diferentes, usariamos 2 capas distintas.
        self.hidden_layer= nn.Sequential(
                                    nn.Linear(input_dimension, 64), #hidden layer
                                    nn.ReLU()
        )
        #Se crean las 2 capas de salida
        self.model1 = nn.Sequential(nn.Linear(64, 10)) #output layer
        self.model2 = nn.Sequential(nn.Linear(64, 10)) #output layer
        # TODO initialize model layers here

    def forward(self, x):
        xf = self.flatten(x)
        # TODO use model layers to predict the two digits
        
        hidden_out = self.hidden_layer(xf)  # Pasar por la capa oculta compartida
        
        # Cada salida usa la misma representación oculta
        out_first_digit = self.model1(hidden_out)
        out_second_digit = self.model2(hidden_out)

        return out_first_digit, out_second_digit
'''
class MLP(nn.Module):

    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        self.model1 = nn.Sequential(
                                    nn.Linear(input_dimension, 64), #hidden layer
                                    nn.ReLU(),
                                    nn.Linear(64, 10), #output layer
        )
        self.model2 = nn.Sequential(
                                    nn.Linear(input_dimension, 64), #hidden layer
                                    nn.ReLU(),
                                    nn.Linear(64, 10), #output layer
        )
        # TODO initialize model layers here

    def forward(self, x):
        xf = self.flatten(x)
        # TODO use model layers to predict the two digits
        
        # Dos salidas para los dos dígitos
        out_first_digit = self.model1(xf)
        out_second_digit = self.model2(xf)

        return out_first_digit, out_second_digit
    
def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = MLP(input_dimension) # TODO add proper layers to MLP class above

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
