"""
UniTN AML final project - 2020/2021
Implementation of Adversarial Discriminative Domain Adaptation - ADDA (Tzeng et al., 2017)
Authors: Andrea Debeni, Stefano Pardini
"""


import os
import argparse
import wandb

import adda.utils as utils
import adda.settings.wandb_settings
import adda.models.arch as arch
import adda.app as app


def main():
    """
    This module is responsible for parsing the command line parameters, launching functions from adda.app package.
    """

    utils.gpu_check()

    parser = argparse.ArgumentParser(description='ADDA')
    parser.add_argument('-phase', type=str, default='1', help='Options: 1 (Pre-training), 2 (Adversarial Adaptation), '
                                                              '3 (Testing)')
    parser.add_argument('-model_arch', type=str, default='LeNetEncoder', help='Options: LeNetEncoder, LeNetClassifier,'
                                                                              'Discriminator, Phase1Model')
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

    # You need to edit settings/wandb_settings.py, specifying WANDB_ENTITY (username), WANDB_API_KEY
    wandb.init(project='AML-ADDA', name='Phase ' + args.phase, group=args.mode)
    wandb.config.epochs = args.e
    wandb.config.batch_size = args.bs

    # In the case you'd like to bypass the args parser:
    # app.phase1_training(epochs=10, batch_size=32)
    app.phase1_test(epochs=10, batch_size=32)
    exit()

    if args.model_arch:
        arch.show_model_arch(args.model_arch, plot=False)
        exit()

    # Phase 1: Pre-training.
    if args.phase == 1:
        if args.mode == 'training':
            app.phase1_training(args.e, args.b)
        elif args.mode == 'test':
            if args.i is None:
                app.phase1_test()
            """
            TODO
            else:
                single_image(args.i)
            """
    # Phase 1: Adversarial Adaptation
    elif args.phase == 2:
        None
        # TODO.

    # Phase 1: Adversarial Adaptation
    elif args.phase == 3:
        None
        # TODO.


if __name__ == '__main__':
    main()
