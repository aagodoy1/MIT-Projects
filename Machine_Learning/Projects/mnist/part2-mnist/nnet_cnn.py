#! /usr/bin/env python

import _pickle as c_pickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model, Flatten

def main():
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # We need to rehape the data back into a 1x28x28 image
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    batch_size = 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification TODO
    model = nn.Sequential(  
              # Cada capa de 3x3 reduce en 2 las dimensiones, ya que tenemos que padding = 0 y stride = 1 (default). 
              # K: Tamaño kernel(filtro)
              # S: Stride(default = 1) cuantos pixeles se mueve hacia el lado cada filtro
              # P: Padding(default = 0) pixeles extra que se agrega para no perder informacion de la imagen. (relleno)
              # Por lo tanto,
              # El calculo es: 
                    # OUT = (IN - K + 2P)/S + 1 
              # En ese caso seria (28-3+2*0)/1 + 1 = 26, por lo que cada filtro 3x3 reduce en 2 el tamaño

              #Entra imagen de 1x28x28
              nn.Conv2d(1, 32, (3, 3)), # Capa convulucional 2d: 1 entrada, 32 feature maps, los filtros tienen tamaño de 3x3 
              # Se convierte en imagen de 32 x 26 x 26
              nn.ReLU(), # Relu
              nn.MaxPool2d((2, 2)), #Max pool de 2x2
              # Se convierte en filtro de 32 x 13 x 13
              nn.Conv2d(32, 64, (3, 3)), # Capa convulucional 2d: 1 entrada, 64 feature maps, los filtros tienen tamaño de 3x3
              # sE convierte en 64 x 11 x 11
              nn.ReLU(), # Relu
              nn.MaxPool2d((2, 2)), #Max pool de 2x2, siempre redondea hacia abajo
              # Se convierte en 64 x 5 x 5
              Flatten(), #Se agrega una flatten layer
              # Deja un vector de 64*5*5
              nn.Linear(64*5*5, 128), 
              nn.Dropout(0.5),
              nn.Linear(128,10)
            )
    ##################################

    train_model(train_batches, dev_batches, model, nesterov=True)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)
    main()
