from os import path
import numpy as np
from scipy.io import loadmat
from adda.settings import config as cfg
from . import DataLoader
from tensorflow.keras.utils import get_file
import os


URL = "http://ufldl.stanford.edu/housenumbers/"
SEED = 31337
SAMPLE_SIZE = 2000
# The number of classes in the SVHN dataset
NUM_CLASSES = 10

class SVHN(DataLoader):

    # SVHN is a real-world image dataset for developing machine learning and object recognition 
    # algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in 
    # flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude 
    # more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real 
    # world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from 
    # house numbers in Google Street View images.

    # 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
    # 73257 digits for training, 26032 digits for testing,

    # Formats:
    #     - Original images with character level bounding boxes. (tar.gz files)
    #     - MNIST-like 32-by-32 images centered around a single character (.mat binaries)
    #         - Here, X is a 4-D matrix containing the images, y is a vector of class labels
    
    
    def __init__(self, sample=True, sample_size=0, normalize=True, resize28=True, download=False):

        if download:
            training_data, training_labels = self.load_data(type="normal", part="train")
            test_data, test_labels = self.load_data(type="normal", part="test")
        else:
            assert os.path.isfile(cfg.SVHN_DATASET_TRAIN_MAT), "SVHN train local not found."
            training_data, training_labels = self.load_data(type="normal", part="train")
            assert os.path.isfile(cfg.SVHN_DATASET_TEST_MAT), "SVHN test local not found."
            test_data, test_labels = self.load_data(type="normal", part="test")


        # np.shape(training_data): (26032, 32, 32, 3)
        # Original image dimensions: (32, 32, 3)

        self.training_data = training_data
        self.training_labels = training_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.num_classes = NUM_CLASSES

        if sample:
            if 0 < sample_size <= len(self.training_data):
                self.SAMPLE_SIZE = sample_size
            else:
                self.SAMPLE_SIZE = SAMPLE_SIZE
            self._sample()

        if resize28:
            self._resize(28,28)
            self.input_shape = self.training_data.shape[1:4]

        if normalize:
            self._normalize()

    
    def load_data(path="svhn_matlab.npz", type="normal", part="train"):

        """Loads the SVHN dataset if not already downloaded.
        # Arguments
            path: path where to cache the dataset locally
                (relative to ~/.keras/datasets).
            type: normal or extra (extra appends ~530K extra images for training)
            part: train or test
        # Returns
            Tuple of Numpy arrays: `(input_train, target_train) or
                                    (input_test, target_test)`, based on the part selected
        Drawn from Tzeng implementation and https://github.com/machinecurve/extra_keras_datasets/blob/master/extra_keras_datasets/svhn.py
        """

        if part=="train":
            dataPath = get_file("{}_train".format(path), origin="{}train_32x32.mat".format(URL))
        else:
            if part=="test":
                dataPath = get_file("{}_test".format(path), origin="{}test_32x32.mat".format(URL))
        
        data = loadmat(dataPath)
        
        # Images are stored in X, labels in y
        images = data['X']
        labels = data['y'].flatten()

        # Casting to 'float32'
        images = images.astype(np.float32)
        images = images.transpose((3,0,1,2))
        labels[labels == 10] = 0
        return images, labels

