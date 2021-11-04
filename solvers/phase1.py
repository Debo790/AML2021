import tensorflow as tf
import random
import numpy as np
import os

"""
The implementation is very similar to the one presented during Lab5 Rota's lesson.
Inside you'll find some useful remarks, in order to better understand the computation flow.
"""

MODEL_PATH = os.getcwd() + '/saved_models/phase1'

class Phase1Solver:
    def __init__(self, batch_size, epochs, ilr=0.0001):
        """
        1. batch_size:
            during the training phase (that is composed by iterations), the model is using a batch
            of the input data with size x: x samples for iteration. The bigger the value, the more statistics have to
            be computed during the weights updating phase. It means that if the batch is quite high, the gradients will be
            quite low; so their updates will be quite small and the training process could take longer time.

        2. epochs:
            how many times you want to use your data before stopping the training.
            Imagine to have a dataset with 100 data points (images), and a batch_size equals to 10.
            It means that after 10 iterations we'll see the entire dataset at least once (then, you'll start again
            from the beginning). Exactly at this point, one epoch is passed. In general, the training phase of a NN is
            measured by epochs.
            Empirical observation: at some time, during the training phase, you'll see that the results won't increase
            anymore. At that point it's better to stop the training.
            So, briefly, the epoch means: how many time the training phase is going to use the entire dataset.

        3. ilr (initial learning rate):
            how fast the training phase is moving toward the local minimum; the higher is the
            value, the faster it'll move to the local minimum, the hardest it'll get close the local minimum.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.initial_learning_rate = ilr

    def train(self, training_data, training_labels, test_data, test_labels, model) -> None:
        """
        In image classification, one of most appropriate loss function is "sparse categorical cross entropy".
        Roughly, it means that the output are converted into probabilities.

        The final layer in our NN produces (also) logits, namely raw prediction values (un-normalized log probabilities).
        SparseCategoricalcrossEntropy(from_logits=True) expects the logits that has not been normalized by softmax.
        """
        supervised_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Here we could use different optimizers; e.g.: stochastic gradient descent, etc.
        # Adam is an adaptive stochastic gradient descent.
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate)

        @tf.function
        # data and labels: batches of 32 elements
        def train_step(data, labels):
            # The tensor dimensions are (data): 32x28x28x1
            # This is the way we're recording the gradients.
            # Reference: https://www.tensorflow.org/guide/autodiff#gradient_tapes
            with tf.GradientTape() as tape:

                # Here we are running the model with the flag training=True.
                # The "call" method (inside the model class) is implicitly invoked.
                logits, preds = model(data, training=True)
                loss = supervised_loss(y_true=labels, y_pred=logits)

            # Here we extract all the gradients computed in the previous 2 lines of code.
            # The trainable variables are all the weights of the NN.
            trainable_vars = model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Optimization
            # Here we apply the gradients to the trainable variables (the variables that have to be trained /
            # improved during the training phase).
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Performance evaluation.
            eq = tf.equal(labels, tf.argmax(preds, -1))
            accuracy = tf.reduce_mean(tf.cast(eq, tf.float32)) * 100

            return loss, accuracy

        @tf.function
        def test_step(data, labels):
            logits, preds = model(data, training=False)
            loss = supervised_loss(y_true=labels, y_pred=logits)
            return loss, logits, preds

        global_step = 0
        best_accuracy = 0.0

        # External for loop: one for each epoch.
        for e in range(self.epochs):

            """
            Shuffling training set
                Why? Because we loaded the data in an ordered fashion, so we'd like to randomize the batches,
                having data points coming from different classes (for each epoch).
                So, variability is good for training. More than having a lot of data, it's important to have
                different data :-)
                So, basically we are forcing the NN to see the data randomized; ok, it's the same data, but the
                order matters.
            """
            perm = np.arange(len(training_labels))
            random.shuffle(perm)
            training_data = training_data[perm]
            training_labels = training_labels[perm]

            # Iteration
            # For each batch size step; it will be: 0, 31, 63, ..
            for i in range(0, len(training_labels), self.batch_size):
                # Here we're slicing the data and the labels.
                data = training_data[i:i + self.batch_size, :]
                labels = training_labels[i:i + self.batch_size, ].astype('int64')
                global_step += 1 # len(labels)

                # Invoke the inner function train_step()
                batch_loss, batch_accuracy = train_step(data, labels)

                # The on going accuracy are shown every 50 steps.
                if global_step % 50 == 0:
                    print('[{0}-{1:03}] loss: {2:0.05}, batch_accuracy: {3:0.03}'.format(
                        e + 1, global_step,
                        batch_loss.numpy(),
                        batch_accuracy.numpy()))
                if global_step == 1:
                    print('number of model parameters {}'.format(model.count_params()))

            # Test the whole test dataset
            # This test phase is performed at the end of each epoch.
            test_preds = tf.zeros((0,), dtype=tf.int64)
            total_loss = list()
            for i in range(0, len(test_labels), self.batch_size):
                data = test_data[i:i + self.batch_size, :]
                labels = test_labels[i:i + self.batch_size, ].astype('int64')
                batch_loss, _, preds = test_step(data, labels)
                batch_preds = tf.argmax(preds, -1)
                test_preds = tf.concat([test_preds, batch_preds], axis=0)
                total_loss.append(batch_loss)
            loss = sum(total_loss) / len(total_loss)
            eq = tf.equal(test_labels, test_preds)
            test_accuracy = tf.reduce_mean(tf.cast(eq, tf.float32)) * 100

            # Save the model if the results are better than the previous trained model.
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                model.save(MODEL_PATH)
            print(
                'End of Epoch {0}/{1:03} -> loss: {2:0.05}, test accuracy: {3:0.03} - best accuracy: {4:0.03}'.format(
                    e + 1, self.epochs,
                    loss.numpy(),
                    test_accuracy.numpy(),
                    best_accuracy))

    def test(self, test_data, test_labels, model, loss_func):

        @tf.function
        def test_step(data, labels):
            logits, preds = model(data, training=False)
            loss = loss_func(y_true=labels, y_pred=logits)
            return loss, logits, preds

        # Forward pass the whole test dataset batch by batch
        test_preds = tf.zeros((0,), dtype=tf.int64)
        total_loss = list()
        for i in range(0, len(test_labels), self.batch_size):
            data = test_data[i:i + self.batch_size, :]
            labels = test_labels[i:i + self.batch_size, ].astype('int64')
            batch_loss, _, preds = test_step(data, labels)
            batch_preds = tf.argmax(preds, -1)
            test_preds = tf.concat([test_preds, batch_preds], axis=0)
            total_loss.append(batch_loss)
        loss = sum(total_loss) / len(total_loss)
        eq = tf.equal(test_labels, test_preds)
        test_accuracy = tf.reduce_mean(tf.cast(eq, tf.float32)) * 100
        return loss, test_accuracy, test_preds
