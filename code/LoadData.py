import numpy
import theano
import theano.tensor as T

class LoadData(object):
    """
    This class loads data
    """

    def __init__(self, link):
        """
        Loads the data

        Input:
        ------
        link: string:
            String to the data
        """
        self.link = link

    def load_data(self):
        """
        This function loads data for the linear regression
        """
        #import pdb; pdb.set_trace()
        data = numpy.load(self.link)
        n = data.shape[1]  // 3
        test_set = data[:,2*n:]
        valid_set = data[:,n:2*n]
        train_set = data[:,0:n]


        def shared_dataset(data_xy, borrow=True):
            """ Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            data_x, data_y = data_xy
            data_x = numpy.array([data_x]).T
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, shared_y

        #import pdb; pdb.set_trace()

        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval
