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

import mlp_extended
reload(mlp_extended)
from mlp_extended import MLP_OPTIMIZATION

from linearRegression_sgd import LinearRegression

link = '/home/fabian/Documents/DeepLearningTutorials/data/SineRegression.npy'

learning_rate = 0.0015
n_epochs = 100000
batch_size = 25
n_hidden = 3
L2_reg = 0.0001


opt = MLP_OPTIMIZATION(learning_rate = learning_rate, n_epochs = n_epochs,
                      batch_size = batch_size, link = link,
                      n_hidden = n_hidden, L2_reg = L2_reg,
                      n_out = 1, output_layer = LinearRegression )
abc = opt.predict()
plt.show(abc)
