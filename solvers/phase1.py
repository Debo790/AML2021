import tensorflow as tf
import random
import numpy as np
import os
import wandb

"""
The implementation is very similar to the ones presented during Lab5/Lab6 Rota's lesson.
Inside you'll find some useful remarks, in order to better understand the computation flow.
"""

MODEL_PATH = os.getcwd() + '/saved_models/phase1'

class Phase1Solver:
    def __init__(self, batch_size, epochs, ilr=0.0001):
        """
        1. batch_size:
            A batch size of 32 means that 32 samples from the training dataset will be used to estimate the error
            gradient before the model weights are updated.

        2. epochs:
            One training epoch means that the learning algorithm has made one pass through the training dataset,
            where examples were separated into randomly selected “batch size” groups.

        3. ilr (initial learning rate):
            The learning rate is a hyperparameter that controls how much to change the model in response to the
            estimated error each time the model weights are updated.
            Choosing the learning rate is challenging as a value too small may result in a long training process
            that could get stuck, whereas a value too large may result in learning a sub-optimal set of weights
            too fast or an unstable training process.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.initial_learning_rate = ilr

    def train(self, training_data, training_labels, test_data, test_labels, model):
        """
        Cross-entropy loss measures the performance of a classification model whose output is a probability value
        between 0 and 1.
        from_logits parameter: "whether y_pred is expected to be a logits tensor. By default, we assume that y_pred
        encodes a probability distribution." (from TensorFlow documentation).
        The final layer in our NN produces (also) logits, namely raw prediction values (un-normalized log probabilities).
        SparseCategoricalcrossEntropy(from_logits=True) expects the logits that has not been normalized by the Softmax
        activation function.
        """
        supervised_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        """
        Defining the type of schedule for the learning rate decay:
            https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay

        This function is going to manage the exponential decay by itself; we just need to pass an initial learning rate,
        also defining after how many steps/iterations the learning rate have to decay (decay_steps=250).
        staircase=True means that the learning rate remains steady until an incremental decay_step is reached  
        (the learning rate it's multiplied by 0.96 every decay_step: 250, 500, 750, etc.).
        
        More generally, this is an optimization effective option: a higher initial learning rate means that the 
        optimization process will quickly move towards the local minimum. Then, subsequently, a smaller learning rate
        will be able to get closer.
        """
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=250,
            decay_rate=0.96,
            staircase=True)
        # Defining the optimizer using the scheduler above
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        @tf.function
        # data and labels: batches of 32 elements
        def train_step(data, labels):
            # The tensor dimensions are (data): 32x28x28x1
            # This is the way we're recording the gradients.
            # Reference: https://www.tensorflow.org/guide/autodiff#gradient_tapes
            with tf.GradientTape() as tape:

                """
                Here we are running the model (fitting the model to data), using the flag training=True.
                It implies that the "call" method inside the model class is implicitly invoked.
                The method returns 2 values: logits (x) and predictions (Softmax output activation function):
                    return x, tf.keras.activations.softmax(x)
                    
                The logits are used in the computation of the supervised loss, together with the true y labels, coming
                from the training data.
                """
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

            # Performance evaluation
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

        # External for loop: one iteration for each epoch.
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
            # For each batch size step (0, 31, 63, ..)
            for i in range(0, len(training_labels), self.batch_size):
                # Here we're slicing the data and the labels.
                data = training_data[i:i + self.batch_size, :]
                labels = training_labels[i:i + self.batch_size, ].astype('int64')
                global_step += 1

                # Invoke the inner function train_step()
                batch_loss, batch_accuracy = train_step(data, labels)

                # The on going accuracy are shown every 50 steps.
                if global_step % 50 == 0:
                    print('[{0}-{1:03}] loss: {2:0.05}, batch_accuracy: {3:0.03}'.format(
                        e + 1, global_step,
                        batch_loss.numpy(),
                        batch_accuracy.numpy()))

                    # Updating the learning rate accord accordingly with the optimizer criteria
                    lr = optimizer._decayed_lr(tf.float32).numpy()

                    wandb.log(
                        {'train/batch_loss': batch_loss,
                         'train/batch_accuracy': batch_accuracy,
                         'train/learning_rate': lr
                         })

                if global_step == 1:
                    print('number of model parameters {}'.format(model.count_params()))

            """
            TEST phase
            Perform test on test set to evaluate the method on unseen data
            """
            loss, test_accuracy, test_preds = self.test(test_data, test_labels, model, supervised_loss)

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

            wandb.log(
                {'test/loss': loss,
                 'test/accuracy': test_accuracy,
                 'test/best_accuracy': best_accuracy,
                 'epoch': e+1})

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
