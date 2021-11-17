import tensorflow as tf
import random
import numpy as np
import wandb
from adda.settings import config


class Phase2Solver:
    """
    Phase2: Adversarial Adaptation

    "Perform adversarial adaptation by learning a target encoder CNN such that a discriminator that sees encoded source and
    target examples cannot reliably predict their domain label." (Tzeng et al., 2017)
    """

    def __init__(self, batch_size, epochs, ilr=0.0001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.initial_learning_rate = ilr

    def train(self, src_training_data, src_training_labels, tgt_training_data, tgt_training_labels,
              src_model, tgt_model, disc_model):

        """
        From TensorFlow documentation:
        "Use this crossentropy loss function when there are two or more label classes. We expect labels to be provided
        in a one_hot representation."
        Reference: https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy

        The aim is to compute the crossentropy loss between labels and predictions (logits).

        The cross entropy of two distributions (real and predicted) that have the same probability distribution will
        always be 0.0. Therefore, a cross-entropy of 0.0 when training a model indicates that the predicted class
        probabilities are identical to the probabilities in the training dataset, e.g. zero loss.
        """

        # TODO.
        # supervised_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        supervised_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        def get_disc_loss(data_labels, pred_labels):
            """
            The aim of the Discriminator is to correctly classify data points coming from the two different data
            distributions (Source and Target).
            The function takes the following parameters as input:
                data_labels: ad-hoc labels associated with source and target data points (1 = source, 0 = target).
                pred_labels: labels predicted by the model (raw output of the final Discriminator NN layer: logits).
            """
            return supervised_loss(data_labels, pred_labels)

        def get_tgt_encoder_loss(data_labels, pred_labels):
            """
            The aim of the Target Encoder is to produce a feature map such that the Discriminator isn't able to
            recognize a data point coming from the Source or the Target dataset.

            "In doing so, we are effectively learning an asymmetric mapping, in which we modify the target model so as
            to match the source distribution". (Tzeng et. al, 2017).

            Data label (real): 1; predicted label: 1; loss = 0
            The Discriminator correctly classify the data point. The Target encoder loses, the Discriminator wins.
            We would like the Discriminator not to be able to discern between the two data distributions.
            So, we have to update the Target NN weights until its outputs are indistinguishable from the Source NN ones.
            Therefore, we have to use the "inverted label GAN loss" as described by Tzeng et al.

            Data label (real): 1 - 1 = 0; predicted label: 1; loss != 0
            In this case, as actually implemented below, Cross Entropy Loss will not be zero; so the updating of the
            target NN weights through backpropagation can start.
            """
            return supervised_loss(1 - data_labels, pred_labels)

        """
        When training a model, it is often useful to lower the learning rate as the training progresses.
        This schedule applies an exponential decay function to an optimizer step.
        """
        disc_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=250,
            decay_rate=0.96,
            staircase=True)
        # Defining the Discriminator optimizer
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr_schedule)

        tgt_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=250,
            decay_rate=0.96,
            staircase=True)
        # Defining the Target optimizer
        tgt_optimizer = tf.keras.optimizers.Adam(learning_rate=tgt_lr_schedule)

        @tf.function
        def train_step(src_data, src_labels, tgt_data, tgt_labels):
            """
            Fitting the models to the data batch.
            In this phase both models are used "as they are" (inference mode; training=False).
            The Source encoder has already been completely trained (Pre-training phase), while the Target encoder
            will be trained afterwards.
            """
            src_features = src_model(src_data, training=False)
            tgt_features = tgt_model(tgt_data, training=False)

            """
            Concatenates the tensors feature; first the Source encoded tensors, then the Target ones.
            The resultant tensor is: (64, 4, 4, 50)
            Should it be 500? If yes, we've to anticipate the flattening operation in the Encoder model.
            """
            concat_features = tf.concat([src_features, tgt_features], 0)

            """
            Prepare real and fake labels:
                1 = Source label
                0 = Target label
            """
            # TODO.
            # int32, float32 or int64?
            # Using int64 we can directly compare the labels with the argmax of the logits; no casting is needed
            src_labels = tf.zeros([tf.shape(src_features)[0]], tf.int64)
            tgt_labels = tf.ones([tf.shape(tgt_features)[0]], tf.int64)
            concat_labels = tf.concat([src_labels, tgt_labels], 0)

            """
            To differentiate automatically, TensorFlow needs to remember what operations happen in what order during
            the forward pass. Then, during the backward pass, TensorFlow traverses this list of operations in reverse
            order to compute gradients. These operations are recorded into a tape.
            """
            with tf.GradientTape() as disc_tape:
                # Fitting the Discriminator to the merged data points (concat)
                disc_logits, disc_preds = disc_model(concat_features, training=True)
                disc_loss = get_disc_loss(concat_labels, disc_logits)

            """
            Discriminator
            """
            # The trainable variables are all the weights of the Discriminator NN
            disc_trainable_vars = disc_model.trainable_variables
            # Backward pass: calculate the gradients by unwrapping the tape
            disc_gradients = disc_tape.gradient(disc_loss, disc_trainable_vars)
            # Optimization: apply the gradients to the trainable variables (weights)
            disc_optimizer.apply_gradients(zip(disc_gradients, disc_trainable_vars))

            # Performance evaluation: simply, the percentage of successfully predicted labels.
            # Training labels VS Discriminator labels

            # disc_preds is a [64 x 2] tensor (the final Discriminator layer has 2 neurons); dtype = float32
            # With tf.argmax(disc_preds, -1)) we get the maximum value among the two, representing the most
            # confident prediction; the resultant dtype = int64
            disc_eq = tf.equal(concat_labels, tf.argmax(disc_preds, -1))
            disc_accuracy = tf.reduce_mean(tf.cast(disc_eq, tf.float32)) * 100

            """
            Target encoder
            """

            """
            To differentiate automatically, TensorFlow needs to remember what operations happen in what order during
            the forward pass. Then, during the backward pass, TensorFlow traverses this list of operations in reverse
            order to compute gradients. These operations are recorded into a tape.
            """
            with tf.GradientTape() as tgt_tape:
                # Fitting the Target encoder to the target data points only
                tgt_features = tgt_model(tgt_data, training=True)
                disc_logits, disc_preds = disc_model(tgt_features, training=False)

                # concat_labels = (64,), int64
                # disc_logits = (64, 2), float32
                tgt_loss = get_tgt_encoder_loss(tgt_labels, disc_logits)

            # The trainable variables are all the weights of the Target NN.
            tgt_trainable_vars = tgt_model.trainable_variables
            # Backward pass: calculate the gradients by unwrapping the tape
            tgt_gradients = tgt_tape.gradient(tgt_loss, tgt_model.trainable_variables)
            # Optimization: apply the gradients to the trainable variables (weights)
            tgt_optimizer.apply_gradients(zip(tgt_gradients, tgt_trainable_vars))

            return disc_loss, disc_accuracy, tgt_loss

        global_step = 0
        best_accuracy = 0.0

        # External for loop: one iteration for each epoch.
        for e in range(self.epochs):

            # Source training set shuffling
            perm = np.arange(len(src_training_labels))
            random.shuffle(perm)
            src_training_data = src_training_data[perm]
            src_training_labels = src_training_labels[perm]

            # Target training set shuffling
            perm = np.arange(len(tgt_training_labels))
            random.shuffle(perm)
            tgt_training_data = tgt_training_data[perm]
            tgt_training_labels = tgt_training_labels[perm]

            # With batch_size = 32, i = [0, 31, 63, ..]
            for i in range(0, len(src_training_labels), self.batch_size):

                # Slicing the data and the labels (both: source and target), generating a batch of size batch_size
                src_data = src_training_data[i:i + self.batch_size, :]
                src_labels = tgt_training_labels[i:i + self.batch_size, ].astype('int64')
                tgt_data = tgt_training_data[i:i + self.batch_size, :]
                tgt_labels = tgt_training_labels[i:i + self.batch_size, ].astype('int64')

                global_step += 1

                # Invoke the inner function train_step()
                batch_loss, batch_accuracy, _ = train_step(src_data, src_labels, tgt_data, tgt_labels)

                # The on going accuracy is shown every 50 steps.
                if global_step % 50 == 0:
                    print('[{0}-{1:03}] batch_loss: {2:0.05}, batch_accuracy: {3:0.03}'
                          .format(e + 1, global_step, batch_loss.numpy(), batch_accuracy.numpy()))

                    # Updating the Discriminator learning rate accordingly with the optimizer criteria
                    disc_lr = disc_optimizer._decayed_lr(tf.float32).numpy()
                    # Updating the Discriminator learning rate accordingly with the optimizer criteria
                    tgt_lr = tgt_optimizer._decayed_lr(tf.float32).numpy()

                    wandb.log(
                        {'train/batch_loss': batch_loss,
                         'train/batch_accuracy': batch_accuracy,
                         'train/disc_learning_rate': disc_lr
                         })

                if global_step == 1:
                    print('Number of Discriminator model parameters {}'.format(disc_model.count_params()))
                    print('Number of Target Encoder model parameters {}'.format(tgt_model.count_params()))

            # TODO.
            # Save the Discriminator and Target model at the end of each epoch
            # Should we evaluate/test, in some way, the Discriminator?
            disc_model.save(config.DISCRIMINATOR_MODEL_PATH)
            tgt_model.save(config.TARGET_MODEL_PATH)

            """
            TODO.
            print('End of Epoch {0}/{1:03} -> loss: {2:0.05}, test accuracy: {3:0.03} - best accuracy: {4:0.03}'
                  .format(e + 1, self.epochs, batch_loss.numpy(), batch_accuracy.numpy(), best_accuracy))

            wandb.log({
                'test/loss': batch_loss,
                'test/accuracy': batch_accuracy,
                'epoch': e + 1
            })
            """