import tensorflow as tf
from tensorflow.keras import Model


class LeNetEncoder(Model):
    """
    Lenet-5 is one of the earliest pre-trained models proposed by Yann LeCun and others in the year 1998,
    in the research paper [1].

    As we can read in Tzeng et al.'s paper:
    "For these experiments, we use the simple modified LeNet architecture provided in the Caffe source code."

    As we can read in [2]:
    "We will use a slightly different version from the original LeNet implementation, replacing the sigmoid activations
    with Rectified Linear Unit (ReLU) activations for the neurons."

    Here the author Tzeng et al. is referring to LeNet-5 NN.

    So, instead of using the original LeNet-5 NN architecture, we are going to use the Caffe's implemented one.
    It only involves using different activation function: ReLU instead of Tanh:

    [1] https://caffe.berkeleyvision.org/gathered/examples/mnist.html
    [2] http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

    Reminder.
    The hyperbolic tangent activation function is also referred to simply as the Tanh (also “tanh” and “TanH“)
    function. It is very similar to the sigmoid activation function and even has the same S-shape.
    The function takes any real value as input and outputs values in the range -1 to 1.

    Some useful resources (mostly LeNet-? implementations):
        - https://github.com/marload/LeNet-keras/blob/master/lenet.py
        - https://github.com/erictzeng/adda/blob/master/adda/models/lenet.py
        - https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
        - https://datahacker.rs/lenet-5-implementation-tensorflow-2-0/
    """

    def __init__(self, input_shape):
        super(LeNetEncoder, self).__init__(trainable=True)

        """
        https://www.researchgate.net/figure/Structure-of-LeNet-5_fig1_312170477
        Input layer: [28 x 28 x 1]
        """
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)

        """        
        Convolutional layer output: [24 x 24 x 20]
        """
        self.conv_layer_1 = tf.keras.layers.Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                                                   padding='valid')

        """        
        Max pooling layer output: [12 x 12 x 20]
        """
        self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        """
        Convolutional layer output: [8 x 8 x 50]
        """
        self.conv_layer_2 = tf.keras.layers.Conv2D(50, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                                                   padding='valid')

        """
        Max pooling layer output: [4 x 4 x 50]
        """
        self.pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        """
        Useful reminder.
        Is it correct to apply ReLU activation function before the max-pooling layer?
        
        MaxPool(Relu(x)) = Relu(MaxPool(x)) for any input. 
        Although it would be technically better first subsample through max-pooling and then apply the non-linearity
        (especially if it is costly, such as the sigmoid), nevertheless in practice it is often done the other way
        round: it doesn't seem to change much in performance.
        
        So, we have to consider ReLU as output activation function.
        """

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.conv_layer_1(x)
        x = self.pool_1(x)
        x = self.conv_layer_2(x)
        x = self.pool_2(x)

        return x

    """
    Thanks to: 
    https://stackoverflow.com/questions/65365745/model-summary-output-is-not-consistent-with-model-definition
    This method overrides the original summary() method that produces non-readable Output Shape information 
    """
    def summary(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='LeNetEncoder')


class LeNetClassifier(Model):
    def __init__(self, output_classes):
        super(LeNetClassifier, self).__init__(trainable=True)

        """
        Input layer: [4 x 4 x 50]
        The shape of the input layer should be exactly the output shape of the last encoder layer (pool_2)
        """
        input_shape = (4, 4, 50)
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)

        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layer
        self.full_layer_1 = tf.keras.layers.Dense(units=500, activation='relu')

        """
        The last layer returns a logits array with length of 10.
        Each node contains a score that indicates the current image belongs to one of the 10 classes.
        
        Here we are implicitly using a linear output activation function
        The linear activation function is also called "identity" (multiplied by 1.0) or "no activation."   
        
        From TensorFlow documentation:
        "activation: Activation function to use. If you don't specify anything, no activation is applied
        (ie. "linear" activation: a(x) = x)."
        """
        self.full_layer_2 = tf.keras.layers.Dense(units=output_classes)

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.flatten(x)
        x = self.full_layer_1(x)
        x = self.full_layer_2(x)

        """
        The output of the NN (x, the logits array) is passed through a Softmax activation function.
        Softmax assigns decimal probabilities to each class in a multi-class problem. Namely it converts the logits to
        probabilities, producing a vector that is non-negative and sums to 1.
        """
        return x, tf.keras.activations.softmax(x)

    # Superclass override
    def summary(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='LeNetClassifier')