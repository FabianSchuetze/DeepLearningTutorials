"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

__docformat__ = 'restructedtext en'

import matplotlib.pyplot as plt
import timeit
import numpy as np
import theano
import theano.tensor as T
from pylab import *
import six.moves.cPickle as pickle

from linearRegression_sgd import LinearRegression
from LoadData import LoadData


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, output_layer):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.nnet.relu
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.LinearRegression = output_layer(
            input = self.hiddenLayer.output,
            n_in = n_hidden,
            n_out = n_out
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.LinearRegression.W ** 2).sum()
        )

        # same holds for the function computing the number of errors
        self.loss = self.LinearRegression.loss

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.LinearRegression.params
        # end-snippet-3

        # keep track of model input
        self.input = input

class MLP_OPTIMIZATION(object):
    """
    A class with the mlp_linear optimization
    """

    def __init__(self, learning_rate, L2_reg, n_epochs, batch_size, link, n_hidden,
                 n_out, output_layer):
        """
        Constructor for the class

        Parameters:
        -----------
        """
        self.learning_rate = learning_rate
        self.L2_reg = L2_reg
        self.n_epochs = n_epochs
        self.link = link
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.output_layer = output_layer

    def __models(self, datasets):
        """
        Returns the models used
        """
        learning_rate, n_hidden = self.learning_rate, self.n_hidden,
        L2_reg, batch_size = self.L2_reg, self.batch_size
        n_out, output_layer = self.n_out, self.output_layer
        index, x, y = T.lscalar(), T.matrix('x'), T.vector('y')
        rng = np.random.RandomState(1234)
        #import pdb; pdb.set_trace()

        x_train, y_train = datasets[0]
        x_validate, y_validate = datasets[1]

        classifier = MLP(
            rng=rng,
            input=x,
            n_in=1,
            n_hidden=n_hidden,
            n_out = n_out,
            output_layer = output_layer
        )

        cost = classifier.loss(y)  + L2_reg * classifier.L2_sqr
        gparams = [T.grad(cost, param) for param in classifier.params]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams)
        ]

        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
            x: x_train[index * batch_size: (index + 1) * batch_size],
            y: y_train[index * batch_size: (index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            inputs=[index],
            outputs=cost,
            givens={
                x: x_validate[index * batch_size:(index + 1) * batch_size],
                y: y_validate[index * batch_size:(index + 1) * batch_size]
            }
        )
        return classifier, train_model, validate_model

    def optimization(self):
        """
        Does the optimization
        """
        batch_size, n_epochs = self.batch_size, self.n_epochs
        data = LoadData(self.link)
        datasets = data.load_data()
        x_train, y_train = datasets[0]
        x_validate, y_validate = datasets[1]


        classifier, train_model, validate_model = self.__models(datasets)

        n_train_batches = x_train.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = x_validate.get_value(borrow=True).shape[0] // batch_size
        done_looping = False
        epoch = 0
        train_loss, validation_loss = [], []
        patience = 500000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                                      # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                      # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
        best_validation_loss = np.inf
        test_score = 0.

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
                #import pdb; pdb.set_trace()
                minibatch_avg_cost = train_model(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    validation_losses = [validate_model(i)
                                         for i in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set

                        # test_losses = [test_model(i)
                        #                for i in range(n_test_batches)]
                        # test_score = np.mean(test_losses)
                        #
                        # print(
                        #     (
                        #         '     epoch %i, minibatch %i/%i, test error of'
                        #         ' best model %f %%'
                        #     ) %
                        #     (
                        #         epoch,
                        #         minibatch_index + 1,
                        #         n_train_batches,
                        #         test_score * 100.
                        #     )
                        # )

                        # save the best model
                        with open('best_model3.pkl', 'wb') as f:
                            pickle.dump(classifier, f)

                if patience <= iter:
                    done_looping = True
                    break


    def predict(self):
        """
        Retruns the forecasts of the run function
        """
        #import pdb; pdb.set_trace()
        #classifier = self.optimization()[1]
        self.optimization()
        classifier = pickle.load(open('best_model3.pkl', 'rb'))
        predict_model = theano.function(
            inputs=[classifier.input],
            outputs=classifier.LinearRegression.y_pred)

        # We can test it on some examples from test test
        data = LoadData(self.link)
        datasets = data.load_data()
        #import pdb; pdb.set_trace()
        x_test, y_test = datasets[2]


        predicted_values = predict_model(x_test.get_value())
        fig = figure()
        _ = plt.scatter(x_test.get_value(), predicted_values, c = 'red', label='Predicted Values')
        _ = plt.scatter(x_test.get_value(), y_test.get_value(), facecolors='none',
                    edgecolors='r', label='Sample Points')
        _ = plt.legend()
        #plt.show()
        return fig
