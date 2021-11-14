import os

# os.getcwd() gets as root directory the same folder in which the main run script (run.py) is located

SAVED_MODELS_PATH = os.getcwd() + '/adda/saved_models/'

SOURCE_MODEL_PATH = SAVED_MODELS_PATH + 'source_encoder'
CLASSIFIER_MODEL_PATH = SAVED_MODELS_PATH + 'classifier'
PHASE1_MODEL_PATH = SAVED_MODELS_PATH + 'phase1'

DISCRIMINATOR_MODEL_PATH = os.getcwd() + 'discriminator'
TARGET_MODEL_PATH = os.getcwd() + 'target_encoder'
