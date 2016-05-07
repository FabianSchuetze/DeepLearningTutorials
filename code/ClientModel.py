import numpy
import theano
import theano.tensor as T
from importlib import reload

import matplotlib.pyplot as plt

theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'None'

from LoadData import LoadData

import PiecewiseLinear_Reinforcement
reload(PiecewiseLinear_Reinforcement)
from PiecewiseLinear_Reinforcement import PiecewiseLinear_Reinforcement

def sgd_optimization(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=20):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """

    link = '/home/fabian/Documents/DeepLearningTutorials/data/DeepQ.npy'
    data = LoadData(link)
    datasets = data.load_data()
    x_train, y_train = datasets[0]
    x_valid, y_valid = datasets[1]

    x = T.matrix('x')
    index = T.lscalar('index')
    y = T.vector('y')

    n_in = 1
    n_out = 1
    batch_size = 20
    import pdb; pdb.set_trace()

    # compute number of minibatches for training, validation and testing
    n_train_batches = x_train.get_value().shape[0] // batch_size
    n_valid_batches = x_valid.get_value().shape[0] // batch_size

    print('... building the model')

    classifier = PiecewiseLinear_Reinforcement(n_in = 1, input = x, n_out = n_out)


    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.loss(y)
    error = classifier.error(y)

    validate_model = theano.function(
        inputs=[index],
        outputs=[cost, error],
        givens={
            x: x_valid[index * batch_size: (index + 1) * batch_size],
            y: y_valid[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=[cost, error],
        updates=updates,
        givens={
            x: x_train[index * batch_size: (index + 1) * batch_size],
            y: y_train[index * batch_size: (index + 1) * batch_size]
        }
    )
    test = [train_model(i) for i in range(n_train_batches)]


    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.005  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)

    best_expected_utility =  -1 * numpy.inf
    test_score = 0.
    #start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)[0]
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                expected_utility = [-1*validate_model(i)[0]
                                     for i in range(n_valid_batches)]
                validation_error = [validate_model(i)[1]
                                    for i in range(n_valid_batches)]
                this_expected_utility = numpy.mean(expected_utility)
                this_validation_error = numpy.mean(validation_error)

                print(
                    'epoch %i, minibatch %i/%i, expected Utility %f validation error  %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_expected_utility,
                        this_validation_error * 100.
                    )
                )

                # if we got the best validation score until now
                if this_expected_utility > best_expected_utility:
                    #improve patience if loss improvement is good enough
                    if this_expected_utility > best_expected_utility *  ( 1 + improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_expected_utility = this_expected_utility
                    # test it on the test set

                    # test_losses = [test_model(i)
                    #                for i in range(n_test_batches)]
                    # test_score = numpy.mean(test_losses)
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
                    # with open('best_model.pkl', 'wb') as f:
                    #
                    #     pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    # print(
    #     (
    #         'Optimization complete with best validation score of %f %%,'
    #         'with test performance %f %%'
    #     )
    #     % (best_validation_loss * 100., test_score * 100.)
    # )
    # print('The code run for %d epochs, with %f epochs/sec' % (
    #     epoch, 1. * epoch / (end_time - start_time)))
    # print(('The code for file ' +
    #        os.path.split(__file__)[1] +
    #        ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

sgd_optimization()
