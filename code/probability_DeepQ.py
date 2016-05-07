"""
Linar Regression
"""
__docformat__ = 'restructedtext en'

import numpy
import theano
import theano.tensor as T

theano.config.optimizer = 'None'


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
        W_values = numpy.asarray(
            numpy.random.uniform(
                low=-numpy.sqrt(6. / (n_in)),
                high=numpy.sqrt(6. / (n_in)),
                size=(n_in)
            ),
            dtype=theano.config.floatX
            )
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        self.b = theano.shared(
            value=numpy.zeros(n_out, dtype=theano.config.floatX),
            name='b', borrow=True
        )

        self.model_values = numpy.array([2, 1, 0.1, 2])
        # self.model_values = theano.shared(
        #     value = numpy.array([2, 1, 0.1, 2], dtype = theano.config.floatX), # wh, wl, r, gamma
        #     name =  'model_values', borrow = True
        # )

        self.y_pred = T.nnet.softmax(T.dot(input, self.W) + self.b[:, None])

        self.input = input

        self.params = [self.W, self.b]

    # def alpha(self):
    #     """
    #     Calculate the paramter alpha
    #     """
    #     input = self.input
    #     y_pred = self.y_pred
    #     model_values = self.model_values
    #     num = (1 - y_pred) * (model_values[1] - model_values[2]*input)
    #     denom = y_pred * (model_values[0] - model_values[2]*input)
    #     return num / denom
    #
    # def choice(self):
    #     """
    #     Return the optimal asset holdings
    #     """
    #     input = self.input
    #     y_pred = self.y_pred
    #     model_values = self.model_values
    #     alpha = self.alpha
    #     num = (T.pow(alpha, model_values[3])  -1) * (model_values[2]* (1 + input))
    #     denom = (model_values[1] - input * model_values[2]
    #             - T.pow(alpha, model_values[3]) * (model_values[0] - input * model_values[2])
    #             )
    #     return num / denom

    def loss(self, y):
        """ The negative of the expected utility

        Parameters:
        ----------

        :y input: array_like:
        :param input: the sample data. Given as the true probabilities associated
        with the input x (here: the price)

        """

        input = self.input
        y_pred = self.y_pred
        model_values = self.model_values
        #import pdb; pdb.set_trace()

        numAlpha = (T.ones_like(y_pred) - y_pred) * (T.ones_like(input) - 0.1 *input)
        denomAlpha = y_pred * (T.ones_like(input) - 0.1 * input)
        alpha = ( (-1 * numAlpha) / denomAlpha) ** 2
        #alpha = 2

        numChoice = alpha * (model_values[2]* (T.ones_like(input) + input))
        denomChoice = (model_values[1] - input * model_values[2]
                - alpha * (model_values[0] - input * model_values[2])
                )
        choice = numChoice / denomChoice

        vH = choice*(model_values[0] - model_values[2] * input) + model_values[2] * ( T.ones_like(input) + input)
        vL = choice*(model_values[1] - model_values[2] * input) + model_values[2] * ( T.ones_like(input) + input)
        uH = T.pow(vH, 1 - model_values[3]) / ( 1 - model_values[3])
        uL = T.pow(vL, 1 - model_values[3])/ ( 1 - model_values[3])
        u = y * uH + ( T.ones_like(y) - y) * uL

        return -T.sum(u)
