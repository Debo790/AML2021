import os

# os.getcwd() gets as root directory the same folder in which the main run script (run.py) is located

SAVED_MODELS_PATH = os.path.join(os.getcwd(), '/saved_models/')
DATASETS_PATH = os.path.join(os.getcwd(), '/datasets/')

SOURCE_MODEL_PATH = os.path.join(SAVED_MODELS_PATH, 'source_encoder')
CLASSIFIER_MODEL_PATH = os.path.join(SAVED_MODELS_PATH, 'classifier')
PHASE1_MODEL_PATH = os.path.join(SAVED_MODELS_PATH, 'phase1')

DISCRIMINATOR_MODEL_PATH = os.path.join(os.getcwd(), 'discriminator')
TARGET_MODEL_PATH = os.getcwd() + 'target_encoder'

MNIST_DATASET_PATH = os.path.join(DATASETS_PATH, 'mnist.npz')
USPS_DATASET_PATH = os.path.join(DATASETS_PATH, 'mnist.npz')