import tensorflow as tf
import os
import cpuinfo
from models import LeNetEncoder, LeNetClassifier, Discriminator
import numpy as np


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


def show_model_architecture(class_name, plot=False):
    """
    Show and/or plot the summary of the selected model architecture
    """
    assert class_name in ['LeNetEncoder', 'LeNetClassifier', 'Discriminator'], 'class_name not found'

    # Dynamically instantiate a object
    if class_name == 'LeNetEncoder':
        # Produce MNIST format input fake data
        data = {
            'training_data': np.ones(shape=(32, 28, 28, 1)),
            'test_data': np.ones(shape=(32, 28, 28, 1)),
            'input_shape': (28, 28, 1)
        }
        model = LeNetEncoder(data['input_shape'])

    else:
        # Produce LeNetEncoder output fake data
        data = {
            'training_data': np.ones(shape=(32, 4, 4, 50)),
            'test_data': np.ones(shape=(32, 4, 4, 50)),
            'input_shape': (4, 4, 50)
        }
        if class_name == 'LeNetClassifier':
            model = LeNetClassifier(output_classes=10)
        elif class_name == 'Discriminator':
            model = Discriminator(data['input_shape'])

    # Prepare a batch of length 32
    training_data = data['training_data'][0:31, :]

    # Fit the model to data
    model(training_data, training=True)

    # Now we can extract and print the model summary
    print(model.summary(data['input_shape']).summary())

    # Eventually graphically
    if plot:
        tf.keras.utils.plot_model(model.summary(data.input_shape), to_file='LeNetEncoder.png', dpi=96, show_shapes=True,
                                  show_layer_names=True, expand_nested=False)