import tensorflow as tf
from tensorflow.keras import Model
from . import LeNetEncoder, LeNetClassifier


class Phase1Model(Model):
    """
    This class represents the model that have to be used during Phase 1 (Pre-training).
    Appends LeNetClassifier to LeNetEncoder.
    """

    def __init__(self, input_shape, output_classes):
        super(Phase1Model, self).__init__(trainable=True)
        self.encoder = LeNetEncoder(input_shape)
        self.classifier = LeNetClassifier(output_classes)

    def call(self, inputs, training=None, mask=None):
        encoder = self.encoder(inputs)
        classifier = self.classifier(encoder)

        # TODO
        # For the moment we're just interested to the classifier output.
        # When we'll have both outputs, we'll have to discriminate them inside the solver.
        # return encoder, classifier
        return classifier

    # Superclass override
    def summary(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='Phase1Model')
