"""
UniTN AML final project - 2020/2021
Implementation of Adversarial Discriminative Domain Adaptation - ADDA (Tzeng et al., 2017)
Authors: Andrea Debeni, Stefano Pardini

This module is the entry point of the program, it is responsible for parsing the command line parameters, launching
functions from adda.app package.
"""

import os
import argparse
import wandb
import cpuinfo
import tensorflow as tf

import adda.settings.wandb_settings
import adda.models.arch as arch
import adda.app as app


def gpu_check():
    """
    Select the computational core: CPU or GPU.
    """
    if tf.test.gpu_device_name():
        print(f'Default GPU device: {tf.test.gpu_device_name()}')
    else:
        CPU_brand = cpuinfo.get_cpu_info()['brand_raw']
        print(f'No GPU found, let\'s use CPU: {CPU_brand}')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main():
    """
    Command line arguments parser; launcher.
    """
    gpu_check()

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
    # app.phase1_test(epochs=10, batch_size=32)
    app.phase2_adaptation(epochs=100, batch_size=32)
    # app.phase3_testing(epochs=10, batch_size=32)
    # arch.show_model_arch('LeNetClassifier', plot=True)
    # exit()

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

    # Phase 2: Adversarial Adaptation
    elif args.phase == 2:
        app.phase2_adaptation(epochs=args.e, batch_size=args.b)

    # Phase 3: Testing
    elif args.phase == 3:
        app.phase3_testing(epochs=args.e, batch_size=args.b)


if __name__ == '__main__':
    main()
