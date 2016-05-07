import numpy
import theano
import theano.tensor as T
from importlib import reload

import matplotlib.pyplot as plt

# import LoadData
# reload(LoadData)
# from LoadData import LoadData
#
# from linearRegression_sgd import LinearRegression

theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'None'

import DeepQ
reload(DeepQ)
from DeepQ import MLP_OPTIMIZATION

import probability_DeepQ
reload(probability_DeepQ)
from probability_DeepQ import LinearRegression

link = '/home/fabian/Documents/DeepLearningTutorials/data/DeepQ.npy'

learning_rate = 0.0015
n_epochs = 1000000
n_hidden = 1
batch_size = 1
L2_reg = 0.0001


opt = MLP_OPTIMIZATION(learning_rate=learning_rate, n_epochs=n_epochs,
                       link=link, batch_size = batch_size,
                       n_hidden=n_hidden, L2_reg=L2_reg,
                       n_out=1, output_layer=LinearRegression)
abc = opt.optimization()
