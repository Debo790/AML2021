import tensorflow as tf
import os
import numpy as np

from adda.settings import config as cfg

SEED = 31337
# As reported in Tzeng et al. paper, from the MNIST dataset are sampled 2000 images
SAMPLE_SIZE = 2000
# The number of classes in the MNIST dataset
NUM_CLASSES = 10


class MNIST:
    """
    Freely drawn from: https://github.com/marload/LeNet-keras/blob/master/data.py

    The MNIST dataset is composed by handwritten digits that contains 60,000 training images and 10,000 testing
    images (28x28x1).

    This class takes care of loading the data and applying some preliminary transformations.
    It is also a container for the data, referenceable through its instance variables.
    """

    def __init__(self, normalize=True, download=False, sample=True):
        """
        Params:
            1. normalize:
                should the [0,255] RGB values be normalized?
            2. download:
                should the dataset be automatically downloaded? If False, it'll be searched in MNIST_DATASET_PATH

        Load training and test dataset
        Dataset dimensions:
            1) train_data = (60000, 28, 28)
            2) train_labels = (60000, )
            3) test_data = (10000, 28, 28)
            4) test_labels = (10000, )
    
        The model expects tensors arranged as (28, 28, 1); we need to apply some transformations.
        """

        # Automatically download the dataset; otherwise use the local one
        if download:
            (training_data, training_labels), (test_data, test_labels) = \
                tf.keras.datasets.mnist.load_data(cfg.MNIST_DATASET_PATH)
        else:
            assert os.path.isfile(cfg.MNIST_DATASET_PATH), 'File not found'
            (training_data, training_labels), (test_data, test_labels) = \
                tf.keras.datasets.mnist.load_data(cfg.MNIST_DATASET_PATH)

        # Image dimensions
        img_rows, img_cols = training_data.shape[1:]

        # Correctly reshape the datasets to be injected into the model
        if tf.keras.backend.image_data_format() == 'channels_first':
            training_data = training_data.reshape(training_data.shape[0], 1, img_rows, img_cols)
            test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            # It should be activated this else branch only
            training_data = training_data.reshape(training_data.shape[0], img_rows, img_cols, 1)
            test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

            """
            # Fake data test
            training_data = np.ones(shape=(60000, 32, 32, 3))
            test_data = np.ones(shape=(10000, 32, 32, 3))
            input_shape = (32, 32, 3)
            """

        # Casting to 'float32' data type
        training_data = training_data.astype('float32')
        test_data = test_data.astype('float32')

        if sample:
            rand = np.random.RandomState(SEED)
            perm = rand.permutation(len(training_data))[:SAMPLE_SIZE]
            perm.sort()
            training_data = training_data[perm]
            training_labels = training_labels[perm]

        if normalize:
            training_data = self._normalize(training_data)
            test_data = self._normalize(test_data)

        # Exposed instance variables
        self.input_shape = input_shape
        self.num_classes = NUM_CLASSES
        self.training_data, self.training_labels = training_data, training_labels
        self.test_data, self.test_labels = test_data, test_labels

    @staticmethod
    def _normalize(data):
        """
        First we divide by 255, getting a value in the range [0,1]; then we subtract 0.5, obtaining a range [-0.5,+0.5].
        The rough idea is to keep the data "near zero", compacting the computation range.
        In particular, here we are imposing a mean equals to zero, in order to reduce the magnitude of the gradients
        and, moreover, the computational effort during the NN training. With higher values the results are going
        to be less accurate.
        return (data / 255) - 0.5
        """

        """"
        Should we alternatively normalize the images in the [-1,1] range, as done in GAN@lab7 example?
        return (data - 127.5) / 127.5
        """

        """
        Otherwise, we could choose to normalize in the the [0,1] range, as in original Tzeng et al. ADDA implementation.
        """
        return data / 255


