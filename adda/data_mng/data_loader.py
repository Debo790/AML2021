import numpy as np
from skimage.transform import resize as skimage_resize

from adda.settings import config as cfg


class DataLoader:
    """
    Parent class that provide uniform methods to the data-wrapper classes.
    """
    def __init__(self):
        # Instance variables defined in the child class
        self.training_data = None
        self.test_data = None
        self.training_labels = None
        self.test_labels = None
        self.num_classes = None
        self.input_shape = None
        self.SAMPLE_SIZE = None

    def gen_fake_data(self, training_size, test_size, shape=(28, 28, 1)):
        """
        For testing purposes only.
        Generate training and test data points.
        """
        self.training_data = np.ones(shape=(training_size, shape))
        self.test_data = np.ones(shape=(test_size, shape))
        self.input_shape = shape

    def _sample(self):
        """
        Carries out a random sampling on both datasets (training and test).
        """
        rand = np.random.RandomState(cfg.SEED)
        perm = rand.permutation(len(self.training_data))[:self.SAMPLE_SIZE]
        perm.sort()

        self.training_data = self.training_data[perm]
        self.training_labels = self.training_labels[perm]

    def _normalize(self):
        """
        We choose to normalize in the the [0,1] range, as in original Tzeng et al. ADDA implementation.
        """
        self.training_data = self.training_data / 255
        self.test_data = self.test_data / 255

    def _resize(self, x=28, y=28):
        """
        It could be useful to resize USPS data points from (16 x 16) to (28 x 28).
        """
        output = np.zeros((self.training_data.shape[0], x, y, 1), dtype=np.float32)
        for i in range(self.training_data.shape[0]):
            output[i, :, :, 0] = skimage_resize(self.training_data[i, :, :, 0], (x, y), mode='constant')
        self.training_data = output

        output = np.zeros((self.test_data.shape[0], x, y, 1), dtype=np.float32)
        for i in range(self.test_data.shape[0]):
            output[i, :, :, 0] = skimage_resize(self.test_data[i, :, :, 0], (x, y), mode='constant')
        self.test_data = output