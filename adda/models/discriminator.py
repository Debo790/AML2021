import tensorflow as tf
from tensorflow.keras import Model

"""
Discriminator model cues from Tzeng et al., page 6:

"When training with ADDA, our adversarial discriminator consists of 3 fully connected layers: two layers with 500
hidden units followed by the final discriminator output. Each of the 500-unit layers uses a ReLU activation function."
"""


class Discriminator(Model):
    def __init__(self, input_shape=(4, 4, 50)):
        super(Discriminator, self).__init__(trainable=True)

        """
        Input layer: [4 x 4 x 50]
        The input layer shape should be equal to the output shape of the last LeNetEncoder layer (pool_2)
        """
        input_shape = (4, 4, 50)
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)

        # Reshape the tensor dimensions to a single one
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layers
        self.full_layer_1 = tf.keras.layers.Dense(units=500, activation='relu')
        self.full_layer_2 = tf.keras.layers.Dense(units=500, activation='relu')

        """
        The last layer doesn't need an activation function (implicitly uses a linear output activation function).
        Tzeng TensorFlow 1.x implementation and PyTorch ADDA implementation both use 2 neurons at the final layer.
        """
        self.full_layer_3 = tf.keras.layers.Dense(units=2)

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.flatten(x)
        x = self.full_layer_1(x)
        x = self.full_layer_2(x)
        x = self.full_layer_3(x)

        """
        The output of the NN is passed through a Softmax activation function.
        Softmax assigns decimal probabilities to each class in a multi-class problem; get the argmax in order to select 
        the most confident class prediction.
        """
        return x, tf.keras.activations.softmax(x)

    # Superclass override
    def summary(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='Discriminator')