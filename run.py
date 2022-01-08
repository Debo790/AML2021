"""
UniTN AML final project - 2020/2021
Implementation of Adversarial Discriminative Domain Adaptation - ADDA (Tzeng et al., 2017)
Authors: Andrea Debeni, Stefano Pardini

This module is the entry point of the program, it is responsible for parsing the command line parameters, launching
functions from adda.app package.
"""

import os
import argparse
import sys

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
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    if tf.test.gpu_device_name():
        print(f'Default GPU device: {tf.test.gpu_device_name()}')
    else:
        CPU_brand = cpuinfo.get_cpu_info()['brand_raw']
        print(f'No GPU found, let\'s use CPU: {CPU_brand}')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def input_check(input: str, role:str):
    
    if input not in ['MNIST', 'USPS', 'SVHN']:
        print("Wrong {}.".format(role))
        sys.exit(0)


def main():
    """
    Command line arguments parser; launcher.
    """

    gpu_check()

    parser = argparse.ArgumentParser(description='ADDA')
    parser.add_argument('-phase', nargs='+', type=str, default='1', help='Options: 1 (Pre-training), 2 (Adversarial Adaptation), '
                                                              '3 (Testing)')
    parser.add_argument('-model_arch', type=str, default='LeNetEncoder', help='Options: LeNetEncoder, LeNetClassifier,'
                                                                              'Discriminator, Phase1Model')
    parser.add_argument('-mode', type=str, default='training', help='Options: training, test')
    parser.add_argument('-sample', type=bool, default=True, help='Using a sampled dataset; default: True')
    parser.add_argument('-wandb', type=str, default='True', help='Log on WandB (default=False)')
    parser.add_argument('-bs', type=int, default=50, help='Batch size')
    parser.add_argument('-e_tr', type=int, default=10, help='Epochs for the training step')
    parser.add_argument('-e_ad', type=int, default=50, help='Epochs for the adaptation step')
    parser.add_argument('-e_te', type=int, default=20, help='Epochs for the test step')
    parser.add_argument('-source', type=str, default='MNIST', help='Options: MNIST, USPS, SVHN')
    parser.add_argument('-target', type=str, default='USPS', help='Options: MNIST, USPS, SVHN')
    
    # TODO
    # We don't have this option yet
    # parser.add_argument('-i', type=str, default=None, help='Path of the image to be classified')

    args = parser.parse_args()
    
    # Specificy source datasets and target datasets for this run
    
    source_ds = args.source
    target_ds = args.target
    input_check(source_ds, "source")
    input_check(target_ds, "target")
    
    input_phases = args.phase

    if args.wandb == 'False' or args.mode in ['test']:
        # If you don't want your script to sync to the cloud
        # https://docs.wandb.ai/guides/track/advanced/environment-variables
        os.environ['WANDB_MODE'] = 'offline'
    elif args.wandb == 'True':
        os.environ['WANDB_MODE'] = 'online'


    # You need to edit settings/wandb_settings.py, specifying WANDB_ENTITY (username), WANDB_API_KEY
    # wandb.init(project='AML-ADDA', name='Phase ' + args.phase, group=args.mode)
    wandb.init(project='AML-ADDA', name='{} -> {}: phase {} complete'.format(source_ds, target_ds, input_phases),
                                 group=args.mode, entity="aml2021")
    wandb.config.epochs_training = args.e_tr
    wandb.config.epochs_adaptation = args.e_ad
    wandb.config.epochs_test = args.e_te
    wandb.config.batch_size = args.bs
    wandb.config.source_dataset = args.source
    wandb.config.target_dataset = args.target
    wandb.config.sampled_dataset = args.sample

    
    sample = args.sample
    batch_size = args.bs
    epochs_tr = args.e_tr
    epochs_ad = args.e_ad
    epochs_te = args.e_te

    # In the case you'd like to bypass the args parser:
    app.phase1_training(epochs=epochs_tr, batch_size=batch_size, source=source_ds, sample=sample)
    
    app.phase1_test(batch_size=batch_size, source=source_ds, target=target_ds, sample=sample)
    
    app.phase2_adaptation(epochs=epochs_ad, batch_size=batch_size, source=source_ds, target=target_ds, sample=sample)

    # app.phase3_testing(epochs=epochs_te, batch_size=batch_size, target=target_ds)
    
    if args.model_arch:
        arch.show_model_arch(args.model_arch, plot=False)
        exit()

    for phase in input_phases:

        # Phase 1: Pre-training.
        if phase == 1:
            if args.mode == 'training':
                app.phase1_training(epochs=epochs_tr, batch_size=batch_size)
            elif args.mode == 'test':
                if args.i is None:
                    app.phase1_test()
                
        # Phase 2: Adversarial Adaptation
        elif phase == 2:
            app.phase2_adaptation(epochs=epochs_ad, batch_size=batch_size)

        # Phase 3: Testing
        elif phase == 3:
            app.phase3_testing(epochs=epochs_te, batch_size=batch_size)
    

if __name__ == '__main__':
    main()
