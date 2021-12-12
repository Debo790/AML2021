import tensorflow as tf
import os

from adda.settings import config as cfg
from . import DataLoader

# As reported in Tzeng et al. paper, from the MNIST dataset are sampled 2000 images
SAMPLE_SIZE = 2000
# The number of classes in the MNIST dataset
NUM_CLASSES = 10


class MNIST(DataLoader):
    """
    Freely drawn from: https://github.com/marload/LeNet-keras/blob/master/data.py

    The MNIST dataset is composed by handwritten digits that contains 60,000 training images and 10,000 testing
    images (28x28x1).

    This class takes care of loading the data and applying some preliminary transformations.
    It is also a container for the data, referenceable through its instance variables.
    """

    def __init__(self, sample=True, sample_size=0, normalize=True, download=False):
        """
        Params:
            1. sample: True/False (it'll be used SAMPLE_SIZE)
            2. sample_size: should SAMPLE_SIZE be override by sample_size?
            3. normalize: should the [0,255] RGB values be normalized?
            4. download: should the dataset be automatically downloaded? If False, it'll be searched in
                MNIST_DATASET_PATH

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

        # Casting to 'float32' data type
        training_data = training_data.astype('float32')
        test_data = test_data.astype('float32')

        # Exposed instance variables
        self.input_shape = input_shape
        self.num_classes = NUM_CLASSES
        self.training_data, self.training_labels = training_data, training_labels
        self.test_data, self.test_labels = test_data, test_labels

        if sample:
            if 0 < sample_size <= len(self.training_data):
                self.SAMPLE_SIZE = sample_size
            else:
                self.SAMPLE_SIZE = SAMPLE_SIZE
            self._sample()

        if normalize:
            self._normalize()
