from adda.data_mng.dataset import Dataset
from tensorflow import keras

from adda.models import LeNetEncoder, LeNetClassifier, Phase1Model, Discriminator
from adda.solvers import Phase1Solver, Phase2Solver, Phase3Solver

from adda.settings import config as cfg


def phase1_training(batch_size, epochs, source, sample):
    """
    Phase 1: Pre-training.
    Training.

    "We first pre-train a source encoder CNN using labeled source image examples." (Tzeng et al., 2017)
    """
    # Load the dataset. We're interested in the whole dataset.
    training_ds = Dataset(source, 'training', sample=sample, batch_size=batch_size)
    test_ds = Dataset(source, 'test', sample=sample, batch_size=batch_size)

    # Load and initialize the model (composed by: encoder + classifier)
    model = Phase1Model(training_ds.get_input_shape(), training_ds.get_num_classes())

    # Instantiate the solver
    solver = Phase1Solver(batch_size, epochs)

    # Run the training
    solver.train(training_ds, test_ds, model)


def phase1_test(batch_size, source, target, sample):
    """
    Phase 1: Pre-training.
    Test.

    Testing the test dataset using the Phase1 saved model.
    """

    # In order to calculate the accuracy "Source only", as reported in the Tzeng et al.'s paper, it is necessary to use
    # the target test set on the source model trained during the previous training phase 1.
    # E.g.: use the USPS test set on the pair model encoder+classifier trained on MNIST.

    # Load the dataset and, contextually, the source model
    src_train_ds = Dataset(source, 'training', sample=sample, batch_size=batch_size)
    # Load the test dataset
    test_ds = Dataset(target, 'test', sample=sample, batch_size=batch_size)

    # Load the trained model
    model = keras.models.load_model(src_train_ds.phase1ModelPath, compile=False)

    # Instantiate the solver
    solver = Phase1Solver(batch_size)

    # Run the test using the test dataset on the source model
    solver.test(test_ds, model)


def phase2_adaptation(batch_size, epochs, source, target, sample):
    """
    Phase 2: Adversarial Adaptation

    "Perform adversarial adaptation by learning a target encoder CNN such that a discriminator that sees encoded source and
    target examples cannot reliably predict their domain label." (Tzeng et al., 2017)
    """

    # Hard-coded constraint: using the whole dataset for SVHN as per Tzeng's paper
    if source == 'SVHN':
        sample = False

    # Load the datasets
    src_training_ds = Dataset(source, 'training', sample=sample, batch_size=batch_size)
    tgt_training_ds = Dataset(target, 'training', sample=sample, batch_size=batch_size)

    # Deal with the fact that the datasets could be of different sizes, causing a not aligned batching.
    # Policy: always choose the bigger dataset as reference point, padding the smaller one.
    src_size = src_training_ds.get_size()
    tgt_size = tgt_training_ds.get_size()

    if src_size > tgt_size:
        tgt_training_ds.pad_dataset(src_size)
    elif tgt_size > src_size:
        src_training_ds.pad_dataset(tgt_size)

    src_model = keras.models.load_model(src_training_ds.sourceModelPath, compile=False)
    """
    As we can read in Tzeng. et al's paper:
    [...] we use the pre-trained source model as an intitialization for the target representation space and
    fix the source (typo: target?) model during adversarial training.
    """
    tgt_model = keras.models.load_model(src_training_ds.sourceModelPath, compile=False)
    # In the case we'd like to use a non-initialized encoder, instead of the source one
    # tgt_model = LeNetEncoder(data_usps.input_shape)

    # Discriminator
    disc_model = Discriminator()
    # Classifier (trained on the source dataset)
    cls_model = keras.models.load_model(src_training_ds.classifierPath, compile=False)

    # Instantiate the solver
    solver = Phase2Solver(batch_size, epochs)

    # Run the training
    solver.train(src_training_ds, tgt_training_ds, src_model, tgt_model, disc_model, cls_model)


def phase3_testing(batch_size, source, target, sample):
    """
    Phase 3: Testing

    "During testing, target images are mapped with the target encoder to the shared feature space and classified by the
    source classifier." (Tzeng et al., 2017)
    """

    # Load the datasets
    src_training_ds = Dataset(source, 'training', sample=sample, batch_size=batch_size)
    tgt_test_ds = Dataset(target, 'training', sample=sample, batch_size=batch_size)

    # Load the Classifier model, trained during phase 1 on the source dataset
    cls_model = keras.models.load_model(src_training_ds.classifierPath, compile=False)
    # Load the Target encoder, trained during phase 2
    tgt_model = keras.models.load_model(tgt_test_ds.targetModelPath, compile=False)

    # Instantiate the solver
    solver = Phase3Solver(batch_size)

    # Run the test
    solver.test(tgt_test_ds, cls_model, tgt_model)