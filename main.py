"""
UniTN AML final project - 2020/2021
Implementation of Adversarial Discriminative Domain Adaptation (ADDA, Tzeng et al.)
Authors: Andrea Debeni, Stefano Pardini
"""

import os
import argparse
import wandb
import tensorflow as tf

from models import Phase1Model
from solvers import Phase1Solver
import utils
from data_loaders import MNIST


def phase1_training():
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
    solver = Phase1Solver(batch_size=32, epochs=10)

    # Run the training
    solver.train(data.training_data, data.training_labels, data.test_data, data.test_labels, model)


def phase1_test():
    """
    Phase 1: Pre-training.
    Test.

    Testing the test dataset using saved model.
    """
    # Load the dataset
    data = MNIST()

    # Load the trained model
    model_path = os.getcwd() + '/saved_models/phase1'
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # Instantiate the solver
    solver = Phase1Solver(batch_size=32, epochs=10)

    supervised_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss, test_accuracy, test_preds = solver.test(data.test_data, data.test_labels, model, supervised_loss)

    print(test_preds)
    print(f'loss: {loss}, accuracy: {test_accuracy}')


if __name__ == '__main__':
    utils.gpu_check()

    parser = argparse.ArgumentParser(description='ADDA')
    parser.add_argument('-phase', type=str, default='1', help='Options: 1, 2, 3')
    parser.add_argument('-mode', type=str, default='training', help='Options: training, test')
    parser.add_argument('-wandb', type=str, default='False', help='Log on WandB (default=False)')
    # TODO
    # parser.add_argument('-i', type=str, default=None, help='Path of the image to be classified')
    args = parser.parse_args()

    # TODO
    # We keep it set to False, due to lack of WandB configuration/tuning
    if args.wandb == 'False' or args.mode == 'deploy':
        # If you don't want your script to sync to the cloud
        # https://docs.wandb.ai/guides/track/advanced/environment-variables
        os.environ['WANDB_MODE'] = 'offline'

    """
    wandb.init(project='AML-ADDA', entity='s-pardox', group=args.mode, name='Phase ' + args.phase)
    wandb.config.epochs = args.e
    wandb.config.batch_size = args.bs
    """

    # Phase 1: pre-training.
    if args.phase == 1:
        if args.mode == 'training':
            phase1_training()
        elif args.mode == 'test':
            if args.i is None:
                phase1_test()
            """
            TODO
            else:
                single_image(args.i)
            """


