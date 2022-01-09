import tensorflow as tf
import wandb

from adda.data_mng import Dataset


class Phase3Solver:
    """
    Phase3 / Testing
    "During testing, target images are mapped with the target encoder to the shared feature space and classified by the
    source classifier." (Tzeng et al., 2017)
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def test(self, test_ds: Dataset, cls_model, tgt_model):
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Single test step: use the model in inference mode to generate predictions on unseen test data
        @tf.function
        def test_step(data, labels):
            # Get the target features using the target model trained during phase 2
            tgt_features = tgt_model(data, training=False)
            # Pass the target features to the classifier trained during phase 1 on the source dataset
            logits, preds = cls_model(tgt_features, training=False)

            loss = loss_function(y_true=labels, y_pred=logits)
            return loss, logits, preds

        # Initialize global containers to store the batch-by-batch results
        whole_preds = tf.zeros((0,), dtype=tf.int64)
        total_loss = list()

        # Reset the iterator position
        test_ds.reset_pos()

        # Forward pass using the whole test dataset, batch by batch
        while test_ds.is_batch_available():
            # Get a data batch (data and labels) composed by batch_size data points
            data_b, labels_b = test_ds.get_batch(padding=False)
            # Invoke the inner function test_step()
            batch_loss, _, preds = test_step(data_b, labels_b)

            # Extract the most confident class prediction (highest probability)
            batch_preds = tf.argmax(preds, -1)
            whole_preds = tf.concat([whole_preds, batch_preds], axis=0)
            total_loss.append(batch_loss)

        loss = sum(total_loss) / len(total_loss)
        # Calculate the number of correctly predicted labels, aggregating the whole batches
        eq = tf.equal(test_ds.labels, whole_preds)
        # Calculate the percentage of successfully predicted labels.
        test_accuracy = tf.reduce_mean(tf.cast(eq, tf.float32)) * 100

        print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+')
        print('Loss: {0:0.05}, test accuracy: {1:0.03}'
              .format(loss.numpy(), test_accuracy.numpy()))
        print('')

        return loss, test_accuracy, whole_preds
