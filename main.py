"""
UniTN AML final project - 2020/2021
Implementation of Adversarial Discriminative Domain Adaptation (ADDA, Tzeng et al.)
Authors: Andrea Debeni, Stefano Pardini

- Phase 1: Pre-training
- Phase 2: Adversarial Adaptation
- Phase 3: Testing
"""

import os
import argparse
import wandb
import tensorflow as tf
from tensorflow import keras

from models import LeNetEncoder, LeNetClassifier, Phase1Model
from solvers import Phase1Solver
from data_loaders import MNIST

import utils
import wandb_settings


def phase1_training(batch_size, epochs):
    """
    Phase 1: Pre-training.
    Training.

    LeNet CNN (LeNet-5):
        1. encoding (mapping of the Source dataset into the Source feature space)
        2. classification
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

    Testing the test dataset using saved model.
    """
    # Load the dataset
    data = MNIST()

    # Load the trained model
    model_path = os.getcwd() + '/saved_models/phase1'
    model = keras.models.load_model(model_path)

    # Instantiate the solver
    solver = Phase1Solver(batch_size, epochs)

    supervised_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss, test_accuracy, test_preds = solver.test(data.test_data, data.test_labels, model, supervised_loss)

    print(test_preds)
    print(f'loss: {loss}, accuracy: {test_accuracy}')


if __name__ == '__main__':
    utils.gpu_check()

    parser = argparse.ArgumentParser(description='ADDA')
    parser.add_argument('-phase', type=str, default='1', help='Options: 1 (Pre-training), 2 (Adversarial Adaptation), '
                                                              '3 (Testing)')
    parser.add_argument('-model_arch', type=str, default='LeNetEncoder', help='Options: LeNetEncoder, LeNetClassifier,'
                                                                              'Discriminator')
    parser.add_argument('-mode', type=str, default='training', help='Options: training, test, deploy')
    parser.add_argument('-wandb', type=str, default='False', help='Log on WandB (default=False)')
    parser.add_argument('-bs', type=int, default=32, help='Batch size')
    parser.add_argument('-e', type=int, default=10, help='Epochs')

    # TODO
    # We don't have this option yet
    # parser.add_argument('-i', type=str, default=None, help='Path of the image to be classified')

    args = parser.parse_args()

    if args.wandb == 'False' or args.mode in ['test', 'deploy']:
        # If you don't want your script to sync to the cloud
        # https://docs.wandb.ai/guides/track/advanced/environment-variables
        os.environ['WANDB_MODE'] = 'offline'

    # You need to edit wandb_settings.py, specifying WANDB_ENTITY (username), WANDB_API_KEY
    wandb.init(project='AML-ADDA', name='Phase ' + args.phase, group=args.mode)
    wandb.config.epochs = args.e
    wandb.config.batch_size = args.bs

    # In the case you'd like to bypass the parser:
    # phase1_training(epochs=10, batch_size=32)
    # phase1_test(epochs=10, batch_size=32)
    # exit()

    if args.model_arch:
        utils.show_model_architecture(args.model_arch, plot=False)
        exit()

    # Phase 1: Pre-training.
    if args.phase == 1:
        if args.mode == 'training':
            phase1_training(args.e, args.b)
        elif args.mode == 'test':
            if args.i is None:
                phase1_test()
            """
            TODO
            else:
                single_image(args.i)
            """


