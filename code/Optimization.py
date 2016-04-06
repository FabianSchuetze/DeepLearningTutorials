import numpy as np

class Optimization(object):
    """
    Training the Layers with Backpropagation
    """
    def __init__(self, batch_size, n_epochs):
        """
        Constructor

        Parameters:
        -----------

        batch_size: scalar(int):
            The size of the mini-batch

        n_epochs: scalar(int):
            The number of epochs (WHAT EXACTLTY?)

        models: tupel with Theano objects
            The functions used for training, validation and testing

        classifier: Theano
            The Actual object (better description needed!)
        """

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        #self.models = models
        #self.classifier = classifier

    def Backpropagation(self, models, datasets):
        """
        General-Purpose function doing Backpropagation.

        Parameters:
        -----------
        dataset
        """
        #import pdb; pdb.set_trace()
        batch_size, n_epochs  = self.batch_size, self.n_epochs

        x_train, y_train = datasets[0]
        x_validate, y_validate = datasets[1]
        x_test, y_test = datasets[2]

        #classifier = self.classifier ## DO I really need this?
        train_model, validate_model, test_model = models

        n_train_batches = x_train.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = x_validate.get_value(borrow=True).shape[0] // batch_size
        done_looping = False
        epoch = 0
        train_loss, validation_loss = [], []
        patience = 50000  # look as this many examples regardless
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
                minibatch_avg_cost = train_model(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    # validation_losses = [validate_model(i)
                    #                      for i in range(n_valid_batches)]
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
# 
                        # test it on the test set
                        test_losses = test_model()
                        # test_losses = [test_model(i)
                        #                for i in range(n_test_batches)]
                        test_score = np.mean(test_losses)

                        print(
                            (
                                '     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_score * 100.
                            )
                        )

                        # save the best model
                        # with open('best_model3.pkl', 'wb') as f:
                        #     pickle.dump(classifier, f)

                if patience <= iter:
                    done_looping = True
                    break
