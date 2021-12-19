import numpy as np

from adda.settings import config as cfg


class Dataset:
    """
    This class takes care of encapsulating one of the objects that encapsulate the datasets (MNIST, USPS, SVHN),
    offering: batching, shuffling, padding, etc.
    """

    def __init__(self, dataset, data_type, sample=True, sample_size=0, batch_size=32):
        """
        Params:
            1. dataset: a string containing one of the following: MNIST, USPS, SVHN
            2. data_type: training or test
            3. sample: as defined in Tzeng's paper, from MNIST are sampled 2000 images, 1800 from USPS.
            4. sample_size: should the built-in SAMPLE_SIZE be override by a user-defined value?
            4. batch_size: slicing size during the batching procedure
        """
        assert dataset in ('MNIST', 'USPS', 'SVHN'), 'Dataset not found'

        if dataset == 'MNIST':
            from .mnist import MNIST
            self.data_obj = MNIST(sample=sample, sample_size=sample_size, normalize=True, download=False)
            self.sourceModelPath = cfg.SOURCE_MODEL_PATH_MNIST
            self.phase1ModelPath = cfg.PHASE1_MODEL_PATH_MNIST
            self.targetModelPath = cfg.TARGET_MODEL_PATH_MNIST

        elif dataset == 'USPS':
            from .usps import USPS
            self.data_obj = USPS(sample=sample, sample_size=sample_size, normalize=True, resize28=True)
            self.sourceModelPath = cfg.SOURCE_MODEL_PATH_USPS
            self.phase1ModelPath = cfg.PHASE1_MODEL_PATH_USPS
            self.targetModelPath = cfg.TARGET_MODEL_PATH_USPS

        elif dataset == 'SVHN':
            from .svhn import SVHN
            self.data_obj = SVHN(sample=True, sample_size=sample_size, normalize=True, resize28=True, download=False)
            self.sourceModelPath = cfg.SOURCE_MODEL_PATH_SVHN
            self.phase1ModelPath = cfg.PHASE1_MODEL_PATH_SVHN
            self.targetModelPath = cfg.TARGET_MODEL_PATH_SVHN

        self.dataset_name = dataset

        self.data = None
        self.labels = None

        assert data_type in ('training', 'test'), 'Data type not found'
        self.data_type = data_type
        self._set_data_type(data_type)

        self.batch_size = batch_size

        # Pointer index in the data structure
        self.pos = 0

    def _set_data_type(self, data_type):
        if self.data_type == 'training':
            self.data = self.data_obj.training_data
            self.labels = self.data_obj.training_labels

        elif self.data_type == 'test':
            self.data = self.data_obj.test_data
            self.labels = self.data_obj.test_labels

    def get_input_shape(self):
        return self.data_obj.input_shape

    def get_num_classes(self):
        return self.data_obj.num_classes

    def get_dataset_name(self):
        return self.dataset_name

    def shuffle(self):
        """
        Randomly rearrange the data points in both data and labels datasets.
        """
        rand = np.random.RandomState(cfg.SEED)
        perm = rand.permutation(len(self.data))
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def pad_dataset(self, size, replace=False):
        """
        Pad the dataset (data and labels) with x random additional data points, reaching 'size'.
        Because of different dataset sample size, this method helps in dealing with a potential batching issue.
        """
        assert size > len(self.data), 'The requested target size is lower than the current one'

        # The number of data points to add
        pad_size = size - len(self.data)
        rnd_choice = np.random.choice(len(self.data), size=pad_size, replace=replace)

        rnd_sample = self.data[rnd_choice]
        self.data = np.concatenate((self.data, rnd_sample))

        rnd_sample = self.labels[rnd_choice]
        self.labels = np.concatenate((self.labels, rnd_sample))

    def _pad_batch(self, data_batch, labels_batch, size, replace=False):
        """
        Pad the batch (data and labels) with x random additional data points, reaching 'size'.
        """
        assert size > len(data_batch), 'The requested target size is lower than the current one'

        # The number of data points to add
        pad_size = size - len(data_batch)
        rnd_choice = np.random.choice(len(self.data), size=pad_size, replace=replace)

        rnd_sample = self.data[rnd_choice]
        data_batch = np.concatenate((data_batch, rnd_sample))

        rnd_sample = self.labels[rnd_choice]
        labels_batch = np.concatenate((labels_batch, rnd_sample))

        return data_batch, labels_batch

    def is_batch_available(self):
        return not self.pos >= len(self.data)

    def get_batch(self, padding=True):
        """
        Return a batch of the chosen size.
        If padding=True, the batch will be automatically padded with randomly sampled data points.
        """
        if self.pos > len(self.data):
            return None, None

        i = self.pos
        bs = self.batch_size

        # Slicing the data and the labels, generating a batch of size batch_size
        data_batch = self.data[i:i + bs, :]
        labels_batch = self.labels[i:i + bs, ].astype('int64')

        # Do we need to pad the current batch?
        if padding and len(data_batch) < bs:
            data_batch, labels_batch = self._pad_batch(data_batch, labels_batch, bs)

        # Next batch position: it could exceed the data length
        self.pos = i + bs

        return data_batch, labels_batch

    def reset_pos(self):
        """
        This method must be called if a new batching iteration is needed.
        """
        self.pos = 0

    def get_pos(self):
        return self.pos

    def get_size(self):
        return len(self.data)
