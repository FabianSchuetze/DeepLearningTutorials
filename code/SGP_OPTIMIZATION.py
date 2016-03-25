import numpy
import theano
import theano.tensor as T
from importlib import reload

import LoadData
reload(LoadData)
from LoadData import LoadData

from linearRegression_sgd import LinearRegression

class SGP_OPTIMIZATION(object):
    """
    This function implements the SGP optimization with a linear function
    """

    def __init__(self, learning_rate, n_epochs, batch_size, link):
        """
        Parameters:
        ----------

        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.link = link

    def __models(self, datasets):
        """
        This function returns the models needed for optimization. The function
        is private because it need not be accessed by the client and is used to
        improve legibility of the code

        Returns:
        --------
        """
        learning_rate, batch_size = self.learning_rate, self.batch_size
        x_train, y_train = datasets[0]
        x_validate, y_validate = datasets[1]
        index, x, y = T.lscalar(), T.matrix(name='x'), T.vector(name='y')
        classifier = LinearRegression(input = x, n_in  = 1)
        cost = classifier.errors(y)

        g_W = T.grad(cost=cost, wrt=classifier.W)
        g_b = T.grad(cost=cost, wrt=classifier.b)
        update = [(classifier.W, classifier.W - learning_rate * g_W),
                   (classifier.b, classifier.b - learning_rate * g_b)]

        #import pdb; pdb.set_trace()
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=update,
            givens={
                x: x_train[index * batch_size: (index + 1) * batch_size],
                y: y_train[index * batch_size: (index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            inputs=[index],
            outputs=cost,
            givens={
                x: x_validate[index * batch_size: (index + 1) * batch_size],
                y: y_validate[index * batch_size: (index + 1) * batch_size]
            }
        )
        return classifier, train_model, validate_model

    def sgd_optimization(self):
        """
        Demonstrate stochastic gradient descent optimization of a linear model

        Parameters:
        -----
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
        """
        batch_size, n_epochs = self.batch_size, self.n_epochs
        data = LoadData(self.link)
        datasets = data.load_data()
        x_train, y_train = datasets[0]
        x_validate, y_validate = datasets[1]

        classifier, train_model, validate_model = self.__models(datasets)

        n_train_batches = x_train.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = x_validate.get_value(borrow=True).shape[0] // batch_size
        epoch = 0
        train_loss, validation_loss = [], []

        while (epoch < n_epochs):
            epoch = epoch + 1
            train_store = [train_model(i) for i in range(n_train_batches)]
            train_loss.append(numpy.mean(train_store))
            validation_store = [validate_model(i) for i in range(n_valid_batches)]
            validation_loss.append(numpy.mean(validation_store))
            print(
                'epoch %i,  validation error %f' %
                (
                    epoch,
                    validation_loss[-1]
                )
            )
        costs = ((train_loss, validation_loss))
        return costs, classifier

#SSE, regressor = sgd_optimization()
