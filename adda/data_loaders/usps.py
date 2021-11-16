import gzip
import numpy as np
from skimage.transform import resize

from adda.settings import config as cfg

SEED = 31337
# As reported in Tzeng et al. paper, from the USPS dataset are sampled 1800 images
SAMPLE_SIZE = 1800
NUM_CLASSES = 10


class USPS:
    """
    Freely drawn from:
        https://github.com/erictzeng/adda/blob/master/adda/data/usps.py
        https://github.com/naoto0804/pytorch-domain-adaptation/blob/master/util/dataset.py

    USPS official reference:
        http://statweb.stanford.edu/~hastie/ElemStatLearn/data.html
    Dataset overview:
        https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.info.txt

    The USPS dataset is composed by handwritten digits: 16x16 grayscale images in the range [0, 1].
    It contains 7291 training observations and 2007 testing observations, distributed as follows:

             0    1   2   3   4   5   6   7   8   9 Total
    Train 1194 1005 731 658 652 556 664 645 542 644 7291
     Test  359  264 198 166 200 160 170 147 166 177 2007

    This class takes care of loading the data and applying some preliminary transformations.
    It is also a container for the data, referenceable through its instance variables.
    """

    def __init__(self, sample=True, normalize=True, resize28=True, zero_centre=True):
        # Training data
        training_data, training_labels = self._read_datafile(cfg.USPS_DATASET_TRAIN)
        # Test data
        test_data, test_labels = self._read_datafile(cfg.USPS_DATASET_TEST)

        if sample:
            rand = np.random.RandomState(SEED)
            perm = rand.permutation(len(training_data))[:SAMPLE_SIZE]
            perm.sort()
            training_data = training_data[perm]
            training_labels = training_labels[perm]

        if resize28:
            training_data = self._resize(training_data)
            test_data = self._resize(test_data)

        if normalize:
            training_data = self._normalize(training_data)
            test_data = self._normalize(training_data)

        if zero_centre:
            training_data = self._normalize(training_data)
            test_data = self._normalize(training_data)

        # training_data.shape: (1800, 16, 16, 1)
        # Image dimensions: (16, 16, 1)
        input_shape = training_data.shape[1:4]

        # Exposed instance variables
        self.input_shape = input_shape
        self.num_classes = NUM_CLASSES
        self.training_data, self.training_labels = training_data, training_labels
        self.test_data, self.test_labels = test_data, test_labels

    @staticmethod
    def _read_datafile(path):
        """
        Read the proprietary USPS digits data file from gzipped archive.
        Freely drawn from Tzeng implementation.
        """
        labels, images = [], []
        with gzip.GzipFile(path) as f:
            for line in f:
                vals = line.strip().split()
                labels.append(float(vals[0]))
                images.append([float(val) for val in vals[1:]])
        labels = np.array(labels, dtype=np.int32)
        labels[labels == 10] = 0
        images = np.array(images, dtype=np.float32).reshape(-1, 16, 16, 1)
        images = (images + 1) / 2
        return images, labels

    @staticmethod
    def _normalize(data):
        """
        Normalize in the the [0,1] range.
        """
        return data / 255

    @staticmethod
    def _resize(data):
        """
        Resize (16 x 16) images in (28 x 28) ones.
        """
        output = np.zeros((data.shape[0], 28, 28, 1), dtype=np.float32)
        for i in range(data.shape[0]):
            output[i, :, :, 0] = resize(data[i, :, :, 0], (28, 28), mode='constant')
        return output

    @staticmethod
    def _zero_centre(data):
        return data * 2.0 - 1.0
