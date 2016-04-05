"""
Linar Regression
"""
__docformat__ = 'restructedtext en'

import numpy
import theano
import theano.tensor as T


class LinearRegression(object):
    """ Calculate Linear Regression """

    def __init__(self, n_in, input, n_out):
        """ Initialize the parameters of the logistic regression

        Parameters:
        -----------

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        """
        self.W = theano.shared(
            value=numpy.zeros(n_in, dtype=theano.config.floatX),
            name='W', borrow=True
        )

        self.b = theano.shared(
            value=numpy.zeros(n_out, dtype=theano.config.floatX),
            name='b', borrow=True
        )

        self.y_pred = T.dot(input, self.W) + self.b[:, None]

        self.input = input

        self.params = [self.W, self.b]

    def errors(self, y):
        """ The squared distance

        Parameters:
        ----------

        :y input: array_like:
        :param input: the sample data

        """
        errors = y- self.y_pred
        return T.sum(T.pow(errors, 2))
