import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = 'C:/Users/user/Documents/GitHub/MIT-Projects/Machine_Learning/Projects/mnist/Datasets/'
#path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        # TODO initialize model layers here
        # Crear capas convulucionales
        self.conv_layers = nn.Sequential(
                                     # Cada capa de 3x3 reduce en 2 las dimensiones, ya que tenemos que padding = 0 y stride = 1 (default). 
                                    # K: Tamaño kernel(filtro)
                                    # S: Stride(default = 1) cuantos pixeles se mueve hacia el lado cada filtro
                                    # P: Padding(default = 0) pixeles extra que se agrega para no perder informacion de la imagen. (relleno)
                                    # Por lo tanto,
                                    # El calculo es: 
                                            # OUT = (IN - K + 2P)/S + 1 
                                    # En ese caso seria (28-3+2*0)/1 + 1 = 26, por lo que cada filtro 3x3 reduce en 2 el tamaño

                                    #Entra imagen de 1x42x28
                                    nn.Conv2d(1, 32, (3, 3)), # Capa convulucional 2d: 1 entrada, 32 feature maps, los filtros tienen tamaño de 3x3 
                                    # Se convierte en imagen de 32 x 40 x 26
                                    nn.ReLU(), # Relu
                                    nn.MaxPool2d((2, 2)), #Max pool de 2x2
                                    # Se convierte en filtro de 32 x 20 x 13
                                    nn.Conv2d(32, 64, (3, 3)), # Capa convulucional 2d: 1 entrada, 64 feature maps, los filtros tienen tamaño de 3x3
                                    # sE convierte en 64 x 18 x 11
                                    nn.ReLU(), # Relu
                                    nn.MaxPool2d((2, 2))
                                    # Se convierte en 64x9x5
        )
        self.flatten_and_hidden =  nn.Sequential(
                                            Flatten(), #Se agrega una flatten layer
                                            # Deja un vector de 64*5*5
                                            nn.Linear(64*9*5, 128), 
                                            nn.ReLU(),
                                            nn.Dropout(0.5)
        )
        self.model1 = nn.Sequential(
                                nn.Linear(128,10) # capa de salida
        )
        self.model2 = nn.Sequential(
                                nn.Linear(128,10) # capa de salida
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten_and_hidden(x)
        out_first_digit = self.model1(x)
        out_second_digit = self.model2(x)
        # TODO use model layers to predict the two digits

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
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

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
