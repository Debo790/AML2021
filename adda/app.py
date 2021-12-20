from adda.data_mng.dataset import Dataset
from tensorflow import keras

from adda.models import LeNetEncoder, LeNetClassifier, Phase1Model, Discriminator
from adda.solvers import Phase1Solver, Phase2Solver, Phase3Solver

from adda.settings import config as cfg

def phase1_training(batch_size, epochs, source):
    """
    Phase 1: Pre-training.
    Training.

    "We first pre-train a source encoder CNN using labeled source image examples." (Tzeng et al., 2017)
    """
    # Load the dataset. We're interested in the whole dataset.
    training_ds = Dataset(source, 'training', sample=False, batch_size=batch_size)
    test_ds = Dataset(source, 'test', sample=False, batch_size=batch_size)

    # Load and initialize the model (composed by: encoder + classifier)
    model = Phase1Model(training_ds.get_input_shape(), training_ds.get_num_classes())

    # Instantiate the solver
    solver = Phase1Solver(batch_size, epochs)

    # Run the training
    solver.train(training_ds, test_ds, model)


def phase1_test(batch_size, epochs, source):
    """
    Phase 1: Pre-training.
    Test.

    Testing the test dataset using the Phase1 saved model.
    """
    # Load the dataset. We're interested in the whole dataset.
    test_ds = Dataset(source, 'test', sample=False, batch_size=batch_size)
    
    # Load the trained model
    model = keras.models.load_model(test_ds.phase1ModelPath, compile=False)
    
    # Instantiate the solver
    solver = Phase1Solver(batch_size, epochs)

    # Run the test
    solver.test(test_ds, model)


def phase2_adaptation(batch_size, epochs, source, target):
    """
    Phase 2: Adversarial Adaptation

    "Perform adversarial adaptation by learning a target encoder CNN such that a discriminator that sees encoded source and
    target examples cannot reliably predict their domain label." (Tzeng et al., 2017)
    """
    # Load the datasets
    src_training_ds = Dataset(source, 'training', sample=True, batch_size=batch_size)
    tgt_training_ds = Dataset(target, 'training', sample=True, batch_size=batch_size)

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
    # tgt_model = LeNetEncoder(data_usps.input_shape)

    disc_model = Discriminator()
    cls_model = keras.models.load_model(cfg.CLASSIFIER_MODEL_PATH, compile=False)

    # Instantiate the solver
    solver = Phase2Solver(batch_size, epochs)

    # Run the training
    solver.train(src_training_ds, tgt_training_ds, src_model, tgt_model, disc_model, cls_model)


def phase3_testing(batch_size, epochs, target):
    """
    Phase 3: Testing

    "During testing, target images are mapped with the target encoder to the shared feature space and classified by the
    source classifier." (Tzeng et al., 2017)
    """

    # Load the dataset
    tgt_training_ds = Dataset(target, 'training', sample=True, batch_size=batch_size)

    # Load the Classifier model, trained during phase 1
    cls_model = keras.models.load_model(cfg.CLASSIFIER_MODEL_PATH)
    # Load the Target encoder, trained during phase 2
    tgt_model = keras.models.load_model(cfg.CLASSIFIER_MODEL_PATH)

    # Instantiate the solver
    solver = Phase3Solver(batch_size, epochs)

    # Run the test
    solver.test(tgt_training_ds, cls_model, tgt_model)
