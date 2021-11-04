import tensorflow as tf
from tensorflow.keras import Model

"""
LeNet is a CNN proposed by Yann LeCun et al. in 1989, composed by 2 macro steps:
    1) encoding,
    2) classification.

Freely drawn from: 
    https://github.com/marload/LeNet-keras/blob/master/lenet.py
    https://github.com/erictzeng/adda/blob/master/adda/models/lenet.py
    https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
"""


class LeNetEncoder(Model):
    """
    In the paper is reported that:
    "For these experiments, we use the simple modified LeNet architecture provided in the Caffe source code."

    As can we read in [1]:
    "We will use a slightly different version from the original LeNet implementation, replacing the sigmoid activations
    with Rectified Linear Unit (ReLU) activations for the neurons."

    So, should we use the original LeNet activation function (tanh, namely a Sigmoid), or should we use ReLU?

    [1] https://caffe.berkeleyvision.org/gathered/examples/mnist.html
    """
    def __init__(self, input_shape):
        super(LeNetEncoder, self).__init__(trainable=True)

        # Input layer: takes as input a 28x28x1 tensor (or 32x32x1 ?)
        self.inputLayer = tf.keras.layers.InputLayer(input_shape=input_shape)

        # Convolutional layer (6 channels output)
        self.conv_layer_1 = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                                   input_shape=input_shape, padding='same')

        # AVG pooling layer
        self.pool_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Convolutional layer (16 channels output)
        self.conv_layer_2 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                                   padding='valid')

        # AVG pooling layer
        self.pool_2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

    def call(self, inputs, training=None, mask=None):
        # The convolutional layers are stacked one after the other
        x = self.inputLayer(inputs)
        x = self.conv_layer_1(x)
        x = self.pool_1(x)
        x = self.conv_layer_2(x)
        x = self.pool_2(x)
        return x


class LeNetClassifier(Model):
    def __init__(self, output_classes):
        super(LeNetClassifier, self).__init__(trainable=True)

        # The shape of the input layer should be exactly the output shape of the last encoder layer.
        input_shape = (5, 5, 16)
        self.inputLayer = tf.keras.layers.InputLayer(input_shape=input_shape)

        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layers
        self.full_layer_1 = tf.keras.layers.Dense(units=120, activation='tanh')
        self.full_layer_2 = tf.keras.layers.Dense(units=84, activation='tanh')
        self.full_layer_3 = tf.keras.layers.Dense(units=output_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.inputLayer(inputs)
        x = self.flatten(x)
        x = self.full_layer_1(x)
        x = self.full_layer_2(x)
        x = self.full_layer_3(x)

        # The output of the NN is passed through a Softmax activation function.
        # Softmax assigns decimal probabilities to each class in a multi-class problem.
        return x, tf.keras.activations.softmax(x)
