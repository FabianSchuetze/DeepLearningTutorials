"""
"""
import os
import sys
import timeit

import numpy

from importlib import reload

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from linearRegression_sgd import LinearRegression
#from Optimization import Optimization
import Optimization
reload(Optimization)
from Optimization import Optimization
from LoadData import LoadData
from mlp import HiddenLayer
from rbm import RBM


# start-snippet-1
class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, output_layer, theano_rng=None, n_ins=1,
                 hidden_layers_sizes=[3, 3], n_outs=1):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.vector('y')  # the labels are presented as 1D vector
                                 # of [int] labels
        # end-snippet-1
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.output_layer = output_layer(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.output_layer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer

        ## self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.output_layer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.In(learning_rate, value=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.errors, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.errors,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score


class DBN_Optimize(object):
    """
    Optimizing the DBN Network above
    """

    def __init__(self, k, finetune_lr, pretraining_epochs, pretrain_lr,
                 training_epochs, batch_size, link):
        """

        Parameters:
        ----------
        :type finetune_lr: float
        :param finetune_lr: learning rate used in the finetune stage
        :type pretraining_epochs: int
        :param pretraining_epochs: number of epoch to do pretraining
        :type pretrain_lr: float
        :param pretrain_lr: learning rate to be used during pre-training
        :type k: int
        :param k: number of Gibbs steps in CD/PCD
        :type training_epochs: int
        :param training_epochs: maximal number of iterations ot run the optimizer
        :type dataset: string
        :param dataset: path the the pickled dataset
        :type batch_size: int
        :param batch_size: the size of a minibatch
        """
        self.finetune_lr = finetune_lr
        self.pretraining_epochs = pretraining_epochs
        self.pretrain_lr = pretrain_lr
        self.k = k
        self.link = link
        self.training_epochs = training_epochs
        self.batch_size = batch_size


    def __pretraining(self, hidden_layers_sizes, pretraining_fns, n_train_batches):
        """
        Pretraining the Model. Private b/c the user doesn't need to see the
        function

        Parameters:
        -----------

        dbn: Theano something
            The dbn class

        train_set_x: Theano something
            The actual training data
        """
        batch_size, k  = self.batch_size, self.k
        print('... getting the pretraining functions')
        print('... pre-training the model')
        start_time = timeit.default_timer()
        ## Pre-train layer-wise
        for i in range(hidden_layers_sizes):
            # go through pretraining epochs
            for epoch in range(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in range(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                                                lr=pretrain_lr))
                print('Pre-training layer %i, epoch %d, cost ' % (i, epoch),)
                print(numpy.mean(c))

        end_time = timeit.default_timer()
        # end-snippet-2
        print('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.),
                              file = sys.stderr)

    def optimize(self):
        """
        Doing the actual optimiziation
        """
        batch_size, finetune_lr = self.batch_size, self.finetune_lr
        batch_size, n_epochs = self.batch_size, self.training_epochs

        data = LoadData(self.link)
        datasets = data.load_data()
        train_set_x = datasets[0][0]
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

        # numpy random generator
        numpy_rng = numpy.random.RandomState(123)

        # construct the Deep Belief Network
        dbn = DBN(numpy_rng=numpy_rng, output_layer = LinearRegression, n_ins=1,
                  hidden_layers_sizes=[3, 3],
                  n_outs=1)

        # Pretraining
        pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size,
                                                    k=k)
        self.__pretraining(dbn.n_layers, pretraining_fns, n_train_batches)

        # Backpropagation
        train_fn, validate_model, test_model = dbn.build_finetune_functions(
            datasets = datasets,
            batch_size = batch_size,
            learning_rate = finetune_lr
        )
        models = (train_fn, validate_model, test_model)
        finetuning = Optimization(batch_size = batch_size, n_epochs = n_epochs)
        finetuning.Backpropagation(models, datasets)

        test  = theano.function(inputs = [dbn.x], outputs = dbn.output_layer.y_pred)
        prediction = test(datasets[2][0].get_value())
        import pdb; pdb.set_trace()
        return dbn

k =1
finetune_lr = 0.0015
pretraining_epochs = 10
pretrain_lr = 0.0015
training_epochs = 5000
batch_size = 25
link = '/home/fabian/Documents/DeepLearningTutorials/data/SineRegression.npy'

if __name__ == '__main__':
    result = DBN_Optimize(k = k,
                          finetune_lr = finetune_lr,
                          pretraining_epochs = pretraining_epochs,
                          training_epochs = training_epochs,
                          batch_size = batch_size,
                          pretrain_lr = pretrain_lr,

                          link = link)
    a, b = result.optimize()
