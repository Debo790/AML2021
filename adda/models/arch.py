from adda.models import LeNetEncoder, LeNetClassifier, Discriminator, Phase1Model
import numpy as np
import tensorflow as tf


def show_model_arch(class_name, plot=False):
    """
    Show and/or plot the summary of the selected model architecture
    """
    assert class_name in ['LeNetEncoder', 'LeNetClassifier', 'Discriminator', 'Phase1Model'], 'class_name not found'

    # Dynamically instantiate a object
    if class_name == 'LeNetEncoder' or class_name == 'Phase1Model':
        # Produce MNIST format input fake data
        data = {
            'training_data': np.ones(shape=(32, 28, 28, 1)),
            'test_data': np.ones(shape=(32, 28, 28, 1)),
            'input_shape': (28, 28, 1)
        }
        if class_name == 'LeNetEncoder':
            model = LeNetEncoder(data['input_shape'])
        elif class_name == 'Phase1Model':
            model = Phase1Model(data['input_shape'], output_classes=10)

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
    print(model.summary(data['input_shape']))

    # Eventually graphically
    if plot:
        tf.keras.utils.plot_model(model.summary(data['input_shape']), to_file=class_name + '.png', dpi=96,
                                  show_shapes=True, show_layer_names=True, expand_nested=False)

    # Recursive call to unwrap the nested models
    if class_name == 'Phase1Model':
        show_model_arch('LeNetEncoder')
        show_model_arch('LeNetClassifier')
