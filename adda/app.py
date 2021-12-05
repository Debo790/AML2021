from tensorflow import keras

from adda.models import LeNetEncoder, LeNetClassifier, Phase1Model, Discriminator
from adda.solvers import Phase1Solver, Phase2Solver, Phase3Solver
from adda.data_loaders import MNIST
from adda.data_loaders import USPS

from adda.settings import config as cfg


def phase1_training(batch_size, epochs):
    """
    Phase 1: Pre-training.
    Training.

    "We first pre-train a source encoder CNN using labeled source image examples." (Tzeng et al., 2017)
    """
    # Load the dataset. We're interested in the whole dataset.
    data = MNIST(sample=False)

    # Load and initialize the model (composed by: encoder + classifier)
    model = Phase1Model(data.input_shape, data.num_classes)

    # Instantiate the solver
    solver = Phase1Solver(batch_size, epochs)

    # Run the training
    solver.train(data.training_data, data.training_labels, data.test_data, data.test_labels, model)


def phase1_test(batch_size, epochs):
    """
    Phase 1: Pre-training.
    Test.

    Testing the test dataset using the Phase1 saved model.
    """
    # Load the dataset
    data = MNIST(sample=False)

    # Load the trained model
    model = keras.models.load_model(cfg.PHASE1_MODEL_PATH, compile=False)

    # Instantiate the solver
    solver = Phase1Solver(batch_size, epochs)

    # Run the test
    solver.test(data.test_data, data.test_labels, model)


def phase2_adaptation(batch_size, epochs):
    """
    Phase 2: Adversarial Adaptation

    "Perform adversarial adaptation by learning a target encoder CNN such that a discriminator that sees encoded source and
    target examples cannot reliably predict their domain label." (Tzeng et al., 2017)
    """
    # Load the datasets
    data_mnist = MNIST(sample=True)
    data_usps = USPS(sample=True, resize28=True)

    src_model = keras.models.load_model(cfg.SOURCE_MODEL_PATH, compile=False)
    """
    As we can read in Tzeng. et al's paper:
    [...] we use the pre-trained source model as an intitialization for the target representation space and
    fix the source (typo: target?) model during adversarial training.
    """
    tgt_model = src_model
    disc_model = Discriminator()
    cls_model = keras.models.load_model(cfg.CLASSIFIER_MODEL_PATH, compile=False)

    # Instantiate the solver
    solver = Phase2Solver(batch_size, epochs)

    # Run the training
    solver.train(data_mnist.training_data, data_mnist.training_labels, data_usps.training_data,
                 data_usps.training_labels, src_model, tgt_model, disc_model, cls_model)


def phase3_testing(batch_size, epochs):
    """
    Phase 3: Testing

    "During testing, target images are mapped with the target encoder to the shared feature space and classified by the
    source classifier." (Tzeng et al., 2017)
    """

    # Load the dataset
    data = USPS(sample=False)

    # Load the Classifier model, trained during phase 1
    cls_model = keras.models.load_model(cfg.CLASSIFIER_MODEL_PATH)
    # Load the Target encoder, trained during phase 2
    tgt_model = keras.models.load_model(cfg.CLASSIFIER_MODEL_PATH)

    # Instantiate the solver
    solver = Phase3Solver(batch_size, epochs)

    # Run the test
    solver.test(data.training_data, data.training_labels, cls_model, tgt_model)






