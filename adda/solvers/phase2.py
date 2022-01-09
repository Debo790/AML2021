import tensorflow as tf
import wandb

from adda.data_mng import Dataset


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

    def train(self, src_training_ds: Dataset, tgt_training_ds: Dataset, src_model, tgt_model, disc_model, cls_model):

        """
        From TensorFlow documentation:
        Use crossentropy loss function when there are two or more label classes. We expect labels to be provided as
        integers. If you want to provide labels using one-hot representation, please use CategoricalCrossentropy loss.
        Reference: https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy

        The aim is to compute the crossentropy loss between labels and predictions (logits).

        The cross entropy of two distributions (real and predicted) that have the same probability distribution will
        always be 0.0. Therefore, a cross-entropy of 0.0 when training a model indicates that the predicted class
        probabilities are identical to the probabilities in the training dataset, e.g. zero loss.
        """
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

            # Some attempts borrowed from different implementations
            # https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
            # return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(data_labels, pred_labels))

            # https://github.com/byeongjokim/ADDA
            # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_labels,
            #                                                               labels=tf.ones_like(pred_labels)))

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

            # Some attempts borrowed from different implementations
            # https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
            # return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(1 - data_labels, pred_labels))

            # https://github.com/byeongjokim/ADDA
            # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_labels,
            #                                                               labels=tf.zeros_like(pred_labels)))

        """
        When training a model, it is often useful to lower the learning rate as the training progresses.
        This schedule applies an exponential decay function to an optimizer step.
        """
        disc_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=250,
            decay_rate=0.96,
            staircase=True)
        # Defining the Discriminator optimizer (different options available)

        # Option 1.
        # disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr_schedule)

        # Option 2.
        # disc_optimizer = tf.keras.optimizers.Adam(self.initial_learning_rate)

        # Option 3.
        # After a series of empirical tests, we have observed that a learning rate equal to 1x10^-6 gives better
        # performance
        disc_optimizer = tf.keras.optimizers.Adam(0.000001)

        tgt_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=250,
            decay_rate=0.96,
            staircase=True)
        # Defining the Target optimizer

        # Option 1.
        # tgt_optimizer = tf.keras.optimizers.Adam(learning_rate=tgt_lr_schedule)

        # Option 2.
        # Also in this case we prefer to use a fixed learning rate equal to 1x10^-4
        tgt_optimizer = tf.keras.optimizers.Adam(self.initial_learning_rate)

        @tf.function
        def train_step(src_data, tgt_data, orig_src_labels, orig_tgt_labels):
            """
            During one step, the program is:
                forward pass, loss calculation, backpropagation, metric updates.

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
            """
            concat_features = tf.concat([src_features, tgt_features], 0)

            """
            Prepare real and fake labels:
                0 = Source label
                1 = Target label
            """
            # int32, float32 or int64?
            # Using int64 we can directly compare the labels with the argmax of the logits; no casting is needed
            src_labels = tf.zeros(len(src_features), tf.int64)
            tgt_labels = tf.ones(len(tgt_features), tf.int64)
            concat_labels = tf.concat([src_labels, tgt_labels], 0)

            #################
            # Discriminator #
            #################

            """
            To differentiate automatically, TensorFlow needs to remember what operations happen in what order during
            the forward pass. Then, during the backward pass, TensorFlow traverses this list of operations in reverse
            order to compute gradients. These operations are recorded into a tape.

            https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
            Calling a model inside a GradientTape scope enables you to retrieve the gradients of the trainable weights
            of the layer with respect to a loss value. Using an optimizer instance, you can use these gradients to
            update these variables (which you can retrieve using model.trainable_weights).
            """
            with tf.GradientTape() as disc_tape:
                # Fitting the Discriminator to the merged data points (concat)
                disc_logits, disc_preds = disc_model(concat_features, training=True)
                disc_loss = get_disc_loss(concat_labels, disc_logits)

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

            ##################
            # Target encoder #
            ##################

            with tf.GradientTape() as tgt_tape:
                # Fitting the Target encoder to the target data points only
                tgt_features = tgt_model(tgt_data, training=True)
                disc_logits, disc_preds = disc_model(tgt_features, training=False)

                # concat_labels = (64,), int64
                # disc_logits = (64, 2), float32
                tgt_loss = get_tgt_encoder_loss(tgt_labels, disc_logits)

            # For debugging purposes only
            # tgt_tape.watched_variables()
            # disc_tape.watched_variables()

            # The trainable variables are all the weights of the Target NN.
            tgt_trainable_vars = tgt_model.trainable_variables
            # Backward pass: calculate the gradients by unwrapping the tape
            tgt_gradients = tgt_tape.gradient(tgt_loss, tgt_trainable_vars)
            # Optimization / weights update: apply the gradients to the trainable variables (weights)
            tgt_optimizer.apply_gradients(zip(tgt_gradients, tgt_trainable_vars))

            disc_eq = tf.equal(tgt_labels, tf.argmax(disc_preds, -1))
            tgt_accuracy = tf.reduce_mean(tf.cast(disc_eq, tf.float32)) * 100

            """
            Classifier performance evaluation: simply, the percentage of successfully predicted labels.
            argmax returns the index with the largest value across axes of a tensor (namely: the most confident 
            class prediction).
            We are bringing into phase 2 what should take place during phase 3. However, we need to observe the 
            adaptation phase by detecting the performance of the target encoder on the classifier.

            The ongoing values are calculated on every batch.

            Remember that cls_preds == logits
            """
            cls_preds, _ = cls_model(src_features, training=False)
            cls_eq = tf.equal(orig_src_labels, tf.argmax(cls_preds, -1))
            src_cls_accuracy = tf.reduce_mean(tf.cast(cls_eq, tf.float32)) * 100

            cls_preds, _ = cls_model(tgt_features, training=False)
            cls_eq = tf.equal(orig_tgt_labels, tf.argmax(cls_preds, -1))
            tgt_cls_accuracy = tf.reduce_mean(tf.cast(cls_eq, tf.float32)) * 100

            cls_preds, _ = cls_model(concat_features, training=False)
            orig_concat_labels = tf.concat([orig_src_labels, orig_tgt_labels], 0)
            cls_eq = tf.equal(orig_concat_labels, tf.argmax(cls_preds, -1))
            concat_cls_accuracy = tf.reduce_mean(tf.cast(cls_eq, tf.float32)) * 100

            return disc_loss, disc_accuracy, tgt_loss, tgt_accuracy, src_cls_accuracy, tgt_cls_accuracy, \
                   concat_cls_accuracy

        global_step = 0

        disc_accuracy_total_list = []
        tgt_accuracy_total_list = []
        tgt_cls_accuracy_total_list = []
        src_cls_accuracy_total_list = []
        concat_cls_accuracy_total_list = []
        # Keep trace of Discriminator worst performance
        disc_accuracy_epoch_worst_mean = 101
        # Keep trace of Target encoder best performance
        tgt_accuracy_epoch_best_mean = 0

        # External for loop: one iteration for each epoch.
        for e in range(self.epochs):

            # Reset the iterator position
            src_training_ds.reset_pos()
            tgt_training_ds.reset_pos()

            # Shuffling the training set
            src_training_ds.shuffle()
            tgt_training_ds.shuffle()

            # In these lists we record the ongoing accuracy for every batch in a single epoch
            disc_accuracy_epoch_list = []
            tgt_accuracy_epoch_list = []
            tgt_cls_accuracy_epoch_list = []
            src_cls_accuracy_epoch_list = []
            concat_cls_accuracy_epoch_list = []

            # Internal loop: one interation for each available batch
            while src_training_ds.is_batch_available() and tgt_training_ds.is_batch_available():
                # Get a data batch (data and labels) composed by batch_size data points
                src_data_b, src_labels_b = src_training_ds.get_batch()
                tgt_data_b, tgt_labels_b = tgt_training_ds.get_batch()

                # +1 every batch
                global_step += 1

                # Invoke the inner function train_step()
                disc_loss, disc_accuracy, tgt_loss, tgt_accuracy, src_cls_accuracy, tgt_cls_accuracy, \
                concat_cls_accuracy = train_step(src_data_b, tgt_data_b, src_labels_b, tgt_labels_b)

                # We keep trace of the performance calculating the following values at the end of each processed batch.
                disc_accuracy_epoch_list.append(disc_accuracy.numpy())
                tgt_accuracy_epoch_list.append(tgt_accuracy.numpy())
                tgt_cls_accuracy_epoch_list.append(tgt_cls_accuracy.numpy())
                src_cls_accuracy_epoch_list.append(src_cls_accuracy.numpy())
                concat_cls_accuracy_epoch_list.append(concat_cls_accuracy.numpy())

                disc_accuracy_total_list.append(disc_accuracy.numpy())
                tgt_accuracy_total_list.append(tgt_accuracy.numpy())
                tgt_cls_accuracy_total_list.append(tgt_cls_accuracy.numpy())
                src_cls_accuracy_total_list.append(src_cls_accuracy.numpy())
                concat_cls_accuracy_total_list.append(concat_cls_accuracy.numpy())

                disc_accuracy_epoch_mean = sum(disc_accuracy_epoch_list) / len(disc_accuracy_epoch_list)
                tgt_accuracy_epoch_mean = sum(tgt_accuracy_epoch_list) / len(tgt_accuracy_epoch_list)
                src_cls_accuracy_epoch_mean = sum(src_cls_accuracy_epoch_list) / len(src_cls_accuracy_epoch_list)
                tgt_cls_accuracy_epoch_mean = sum(tgt_cls_accuracy_epoch_list) / len(tgt_cls_accuracy_epoch_list)
                concat_cls_accuracy_epoch_mean = sum(concat_cls_accuracy_epoch_list) / \
                                                 len(concat_cls_accuracy_epoch_list)

                disc_accuracy_total_mean = sum(disc_accuracy_total_list) / len(disc_accuracy_total_list)
                tgt_accuracy_total_mean = sum(tgt_accuracy_total_list) / len(tgt_accuracy_total_list)
                src_cls_accuracy_total_mean = sum(src_cls_accuracy_total_list) / \
                                                 len(src_cls_accuracy_total_list)
                tgt_cls_accuracy_total_mean = sum(tgt_cls_accuracy_total_list) / \
                                                 len(tgt_cls_accuracy_total_list)
                concat_cls_accuracy_total_mean = sum(concat_cls_accuracy_total_list) / \
                                                    len(concat_cls_accuracy_total_list)

                # Ongoing accuracy is shown every 50 processed batches
                if global_step % 50 == 0:
                    print('')
                    print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+='
                          '+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+')
                    print('[{0}-{1:03}] disc_loss: {2:0.05}, disc_accuracy: {3:0.03}, tgt_loss: {4:0.03}, '
                          'tgt_accuracy: {5:0.05}, src_cls_accuracy: {6:0.03}, tgt_cls_accuracy: {7:0.03}, '
                          'concat_cls_accuracy: {8:0.03}'
                          .format(e + 1, global_step, disc_loss.numpy(), disc_accuracy.numpy(), tgt_loss.numpy(),
                                  tgt_accuracy.numpy(), src_cls_accuracy.numpy(), tgt_cls_accuracy.numpy(),
                                  concat_cls_accuracy.numpy()))
                    print('')
                    print('disc_accuracy_epoch_mean: {0:0.03}'.format(disc_accuracy_epoch_mean))
                    print('tgt_accuracy_epoch_mean: {0:0.03}'.format(tgt_accuracy_epoch_mean))
                    print('src_cls_accuracy_epoch_mean: {0:0.03}'.format(src_cls_accuracy_epoch_mean))
                    print('tgt_cls_accuracy_epoch_mean: {0:0.03}'.format(tgt_cls_accuracy_epoch_mean))
                    print('concat_cls_accuracy_epoch_mean: {0:0.03}'.format(concat_cls_accuracy_epoch_mean))
                    print('')
                    print('disc_accuracy_total_mean: {0:0.03}'.format(disc_accuracy_total_mean))
                    print('tgt_accuracy_total_mean: {0:0.03}'.format(tgt_accuracy_total_mean))
                    print('src_cls_accuracy_total_mean: {0:0.03}'.format(src_cls_accuracy_total_mean))
                    print('tgt_cls_accuracy_total_mean: {0:0.03}'.format(tgt_cls_accuracy_total_mean))
                    print('concat_cls_accuracy_total_mean: {0:0.03}'.format(concat_cls_accuracy_total_mean))

                if global_step == 1:
                    print('Number of Discriminator model parameters {}'.format(disc_model.count_params()))
                    print('Number of Target Encoder model parameters {}'.format(tgt_model.count_params()))

            # Extracting the learning rate value from the optimizers
            disc_lr = disc_optimizer._decayed_lr(tf.float32).numpy()
            tgt_lr = tgt_optimizer._decayed_lr(tf.float32).numpy()

            # Here we're going to push to WanDB the learning ongoing results every epoch
            wandb.log({
                'train/disc_loss': disc_loss,
                'train/disc_accuracy': disc_accuracy,
                'train/tgt_loss': tgt_loss,
                'train/tgt_accuracy': tgt_accuracy,
                'train/disc_learning_rate': disc_lr,
                'train/tgt_learning_rate': tgt_lr,

                'train/disc_accuracy_epoch_mean': disc_accuracy_epoch_mean,
                'train/tgt_accuracy_epoch_mean': tgt_accuracy_epoch_mean,
                'train/src_cls_accuracy_epoch_mean': src_cls_accuracy_epoch_mean,
                'train/tgt_cls_accuracy_epoch_mean': tgt_cls_accuracy_epoch_mean,
                'train/concat_cls_accuracy_epoch_mean': concat_cls_accuracy_epoch_mean,

                'train/disc_accuracy_total_mean': disc_accuracy_total_mean,
                'train/tgt_accuracy_total_mean': tgt_accuracy_total_mean,
                'train/src_cls_accuracy_total_mean': src_cls_accuracy_total_mean,
                'train/tgt_cls_accuracy_total_mean': tgt_cls_accuracy_total_mean,
                'train/concat_cls_accuracy_total_mean': concat_cls_accuracy_total_mean,
            })

            # Save the Target model if the results are better than the previous
            if disc_accuracy_epoch_mean < disc_accuracy_epoch_worst_mean:
                disc_accuracy_epoch_worst_mean = disc_accuracy_epoch_mean
                # Save the Target model
                tgt_model.save(tgt_training_ds.targetModelPath, save_format='tf')

        print('=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+='
              '+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+')
        print('')