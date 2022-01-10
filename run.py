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


def input_check(input: str, role: str):
    """
    Verify the correctness of the selected dataset
    """
    if input not in ['MNIST', 'USPS', 'SVHN']:
        print("Wrong {}.".format(role))
        sys.exit(0)


def main():
    """
    Command line arguments parser, WanDB initializer, launcher.
    """

    gpu_check()

    parser = argparse.ArgumentParser(description='ADDA')
    parser.add_argument('-phase', nargs='+', type=str, default='1', help='Options: 1 (Pre-training), '
                                                                         '2 (Adversarial Adaptation), 3 (Testing)')
    parser.add_argument('-model_arch', type=str, default='LeNetEncoder', help='Options: LeNetEncoder, LeNetClassifier,'
                                                                              'Discriminator, Phase1Model')
    parser.add_argument('-sample_tr', type=bool, default=False, help='Using a sampled dataset during the Training '
                                                                     'phase; default: False')
    parser.add_argument('-sample_ad', type=bool, default=True, help='Using a sampled dataset during the Adaptation '
                                                                    'phase; default: True')
    parser.add_argument('-sample_te', type=bool, default=False, help='Using a sampled dataset during the Test '
                                                                     'phase; default: False')
    parser.add_argument('-wandb', type=str, default='True', help='Log on WandB (default=False)')
    parser.add_argument('-bs', type=int, default=50, help='Batch size')
    parser.add_argument('-e_tr', type=int, default=10, help='Epochs for the Training phase')
    parser.add_argument('-e_ad', type=int, default=50, help='Epochs for the Adaptation phase')
    parser.add_argument('-source', type=str, default='MNIST', help='Options: MNIST, USPS, SVHN')
    parser.add_argument('-target', type=str, default='USPS', help='Options: MNIST, USPS, SVHN')

    args = parser.parse_args()

    # Specific source datasets and target datasets for this run
    source_ds = args.source
    target_ds = args.target
    input_check(source_ds, "source")
    input_check(target_ds, "target")

    input_phases = args.phase

    if args.wandb == 'False':
        # If you don't want your script to sync to the cloud
        # https://docs.wandb.ai/guides/track/advanced/environment-variables
        os.environ['WANDB_MODE'] = 'offline'
    elif args.wandb == 'True':
        os.environ['WANDB_MODE'] = 'online'

    # You need to edit settings/wandb_settings.py, specifying WANDB_ENTITY (username), WANDB_API_KEY
    wandb.init(project='AML-ADDA', name='{} -> {}: phase {} * morning run'.format(source_ds, target_ds, input_phases),
               entity="aml2021", group="final")
    # wandb.init(project='AML-ADDA', name='{} -> {}: phase {} complete'.format(source_ds, target_ds, input_phases),
    #            entity="aml2021")

    # WanDB options
    wandb.config.epochs_training = args.e_tr
    wandb.config.epochs_adaptation = args.e_ad
    wandb.config.batch_size = args.bs
    wandb.config.source_dataset = args.source
    wandb.config.target_dataset = args.target
    wandb.config.sampled_training_dataset = args.sample_tr
    wandb.config.sampled_adaptation_dataset = args.sample_ad
    wandb.config.sampled_test_dataset = args.sample_te

    # Local variables from args variables
    sample_tr = args.sample_tr
    sample_ad = args.sample_ad
    sample_te = args.sample_te
    batch_size = args.bs
    epochs_tr = args.e_tr
    epochs_ad = args.e_ad

    # Decomment the following lines in the case you'd like to bypass the args parser.
    ##### START #####
    """
    sample_tr = False
    epochs_tr = 10
    source_ds = 'USPS'
    epochs_ad = 50
    target_ds = 'MNIST'
    #
    app.phase1_training(epochs=epochs_tr, batch_size=batch_size, source=source_ds, sample=sample_tr)
    # Test accuracy on source test set
    app.phase1_test(batch_size=batch_size, source=source_ds, target=source_ds, sample=sample_te)
    # "Source only" accuracy (test accuracy on target test set)
    app.phase1_test(batch_size=batch_size, source=source_ds, target=target_ds, sample=sample_te)
    app.phase2_adaptation(epochs=epochs_ad, batch_size=batch_size, source=source_ds, target=target_ds, sample=sample_ad)
    app.phase3_testing(batch_size=batch_size, source=source_ds, target=target_ds, sample=sample_te)
    exit()
    """
    ##### END #####

    if args.model_arch:
        # This option simply produces an image with the selected network architecture
        arch.show_model_arch(args.model_arch, plot=False)

    for phase in input_phases:

        # Phase 1: Pre-training.
        if phase == '1':
            app.phase1_training(epochs=epochs_tr, batch_size=batch_size, source=source_ds, sample=sample_tr)
            # Test accuracy on source test set, using the source encoder
            app.phase1_test(batch_size=batch_size, source=source_ds, target=source_ds, sample=sample_te)
            # "Source only" accuracy (test accuracy on target test set), using the source encoder
            app.phase1_test(batch_size=batch_size, source=source_ds, target=target_ds, sample=sample_te)

        # Phase 2: Adversarial Adaptation
        elif phase == '2':
            app.phase2_adaptation(epochs=epochs_ad, batch_size=batch_size, source=source_ds, target=target_ds,
                                  sample=sample_ad)

        # Phase 3: Testing
        elif phase == '3':
            app.phase3_testing(batch_size=batch_size, source=source_ds, target=target_ds, sample=sample_te)


if __name__ == '__main__':
    main()
