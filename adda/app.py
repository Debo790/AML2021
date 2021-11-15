from tensorflow import keras

from adda.models import LeNetEncoder, LeNetClassifier, Phase1Model
from adda.solvers import Phase1Solver
from adda.data import MNIST

from adda.settings import config as cfg


def phase1_training(batch_size, epochs):
    """
    Phase 1: Pre-training.
    Training.

    "We first pre-train a source encoder CNN using labeled source image examples." (Tzeng et al., 2017)
    """
    # Load the dataset
    data = MNIST()

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
    data = MNIST()

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
    # TODO.


def phase3_testing(batch_size, epochs):
    """
    Phase 3: Testing

    "During testing, target images are mapped with the target encoder to the shared feature space and classified by the
    source classifier." (Tzeng et al., 2017)
    """
    # TODO.
