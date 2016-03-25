import numpy
import theano
import theano.tensor as T
from importlib import reload

# import LoadData
# reload(LoadData)
# from LoadData import LoadData
#
# from linearRegression_sgd import LinearRegression

import SGP_OPTIMIZATION
reload(SGP_OPTIMIZATION)
from SGP_OPTIMIZATION import SGP_OPTIMIZATION

link = '/home/fabian/Documents/DeepLearningTutorials/data/LinearRegression.npy'


opt = SGP_OPTIMIZATION(learning_rate = 0.0013, n_epochs = 100, batch_size = 15, link = link)

SSE, classifier = opt.sgd_optimization()
