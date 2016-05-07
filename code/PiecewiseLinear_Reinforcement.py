"""
Piecwise Linear Probability Function.

This class calculates the Piecwise linear probabiltiy. I use it to calculate
the probabiltiy for a high return.  The loss function is a uitlity funciton
stating how good a probability prediction will perform in the long run.

"""
__docformat__ = 'restructedtext en'

import numpy
import theano
import theano.tensor as T

#theano.config.optimizer = 'None'

class PiecewiseLinear_Reinforcement(object):
    """
    Piecwise Lienar Probability Function with Evaluation in terms of utilities
    """

    def __init__(self, n_in, input, n_out):
        """ Initialize the parameters of the logistic regression

        Parameters:
        -----------
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        """
        W_values = numpy.asarray(
            numpy.random.uniform(low=0, high=1 / 3, size=(n_in, )),
            dtype=theano.config.floatX
        )
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        self.b = theano.shared(
            value=numpy.zeros(n_out, dtype=theano.config.floatX),
            name='b', borrow=True
        )

        self.model_values = numpy.array([2, 1, 0.1, 2])

        linear = T.dot(input, self.W) + self.b[:, None]

        self.y_pred =  linear

        self.input = input

        self.params = [self.W, self.b]

    def alpha(self):
        """
        The constant used in the optimal choice function, as funciton of the
        forecast value

        Returns:
        --------
        a: Theano.Tensor Tpye:
            The parameter alpha used in the coice function
        """
        x = self.input
        y_pred = self.y_pred
        wh, wl, r, gamma = self.model_values
        numAlpha = (1 - y_pred) * (wl - r * x[:, 0])
        denomAlpha = y_pred * (wh - r * x[:, 0])
        a = (-1 * numAlpha / denomAlpha) ** (gamma)
        return a

    def choice(self):
        """
        Return the optimal asset holdings

        Returns:
        --------
        c: Theano.tensor.type
            The asset demand function
        """
        x = self.input
        wh, wl, r, gamma = self.model_values
        a = self.alpha()
        numChoice = r * (1 + x[:, 0]) * (1 - a)
        denomChoice = (wl - r * x[:, 0]) - a * (wh - r * x[:, 0])
        c =  numChoice / denomChoice
        return c

    def utilities(self):
        """
        Calculates the utilities in each state

        Returns:
        -------
        uH, uL: Theano.tensor.Type
            The utility for state high and state low
        """

        x = self.input
        c = self.choice()
        wh, wl, r, gamma = self.model_values
        vH = c * (wh - r * x[:, 0]) + r * (1 + x[:, 0])
        vL = c * (wl - r * x[:, 0]) + r * (1 + x[:, 0])
        uH = vH**(1 - gamma) / (1 - gamma)
        uL = vL**(1 - gamma) / (1 - gamma)
        return uH, uL

    def error(self, y):
        """
        This function computes the squared forecast error. The squared
        forecast error is (commonly) not used as loss function.

        Parameters:
        -----------
        y: theano.tensor.TensorType:
            The true probabilities of each state

        Returns:
        --------
        error: theano.tensor.var.TensorVariable
            The sum squared forecast error
        """
        y_pred = self.y_pred
        e = y - y_pred
        return T.sum(T.pow(e, 2))

    def loss(self, y):
        """
        The negative of the expected utilit function

        Parameters:
        ----------
        y: Theano.tensor.TensorType
            The true probabilities of each state. Used to calculate the long run
            utility

        Returns:
        --------
        l: Theano.tensor.TensorType
            The negative expected utility
        """
        uH, uL = self.utilities()
        l =  -1 * T.sum(y * uH + (1 - y) * uL)
        return l
