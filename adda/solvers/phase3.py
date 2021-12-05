import tensorflow as tf
import wandb
from adda.settings import config


class Phase3Solver:
    """
    Phase3 / Testing
    "During testing, target images are mapped with the target encoder to the shared feature space and classified by the
    source classifier." (Tzeng et al., 2017)
    """

    def __init__(self, batch_size, epochs, ilr=0.0001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.initial_learning_rate = ilr

    def test(self, test_data, test_labels, cls_model, tgt_model):
        """
        Cross-entropy loss measures the performance of a classification model whose output is a probability value
        between 0 and 1.

        from_logits parameter: "whether y_pred is expected to be a logits tensor. By default, we assume that y_pred
        encodes a probability distribution." (from TensorFlow documentation).

        The final layer in our NN produces (also) logits, namely raw prediction values (un-normalized log
        probabilities).
        SparseCategoricalcrossEntropy(from_logits=True) expects the logits that has not been normalized by the Softmax
        activation function.
        """
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # @tf.function
        def test_step(data, labels):
            tgt_features = tgt_model(data, training=False)
            logits, preds = cls_model(tgt_features, training=False)
            loss = loss_function(y_true=labels, y_pred=logits)
            return loss, logits, preds

        # Initialize global containers to store the batch-by-batch results
        test_preds = tf.zeros((0,), dtype=tf.int64)
        total_loss = list()

        # Forward pass using the whole test dataset, batch by batch
        for i in range(0, len(test_labels), self.batch_size):
            # Slicing the data and the labels
            data = test_data[i:i + self.batch_size, :]
            labels = test_labels[i:i + self.batch_size, ].astype('int64')

            batch_loss, _, preds = test_step(data, labels)

            # Extract the most confident class prediction (highest probability)
            batch_preds = tf.argmax(preds, -1)
            test_preds = tf.concat([test_preds, batch_preds], axis=0)
            total_loss.append(batch_loss)

        loss = sum(total_loss) / len(total_loss)
        # Calculate the number of correctly predicted labels, aggregating the whole batches
        eq = tf.equal(test_labels, test_preds)
        # Calculate the the percentage of successfully predicted labels.
        test_accuracy = tf.reduce_mean(tf.cast(eq, tf.float32)) * 100

        print('Loss: {0:0.05}, test accuracy: {1:0.03}'
              .format(loss.numpy(), test_accuracy.numpy()))

        return loss, test_accuracy, test_preds
