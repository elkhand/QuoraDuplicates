from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

from model import Model

class BOWModel(Model):
    """
    Implements a bag-of-words model.
    """
    def add_placeholders(self):
        self.input1_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.input2_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None,))
        self.seqlen1_placeholder = tf.placeholder(tf.int32, (None,))
        self.seqlen2_placeholder = tf.placeholder(tf.int32, (None,))
        self.featmask1_placeholder = tf.placeholder(tf.float32, (None, self.max_length))
        self.featmask2_placeholder = tf.placeholder(tf.float32, (None, self.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, [])

    def create_feed_dict(self, inputs1_batch, inputs2_batch, seqlen1_batch, seqlen2_batch, featmask1_batch, featmask2_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict.

        Note: The signature of this function must match the return value of preprocess_sequence_data.

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.input1_placeholder: inputs1_batch,
            self.input2_placeholder: inputs2_batch,
            self.seqlen1_placeholder: seqlen1_batch,
            self.seqlen2_placeholder: seqlen2_batch,
            self.featmask1_placeholder: featmask1_batch,
            self.featmask2_placeholder: featmask2_batch,
            self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            feed_dict.update({self.labels_placeholder: np.array(labels_batch, dtype=np.float32)})
        return feed_dict

    def add_prediction_op(self):
        # embedding with mask
        x1 = self.add_embedding(1) * tf.expand_dims(self.featmask1_placeholder, 2) # (?, L, embed_size) .* (?, L)
        x2 = self.add_embedding(2) * tf.expand_dims(self.featmask2_placeholder, 2)
        dropout_rate = self.dropout_placeholder

        # bag of words mean and max over embeddings
        z1 = tf.reduce_sum(x1, 1) / tf.expand_dims(tf.reduce_sum(self.featmask1_placeholder, 1), 1) # (?, embed_size)
        z2 = tf.reduce_sum(x2, 1) / tf.expand_dims(tf.reduce_sum(self.featmask1_placeholder, 1), 1)
        x1 = tf.reduce_max(x1, 1)
        x2 = tf.reduce_max(x2, 1)

        # initialize variables
        xavier_init = tf.contrib.layers.xavier_initializer()
        W1 = tf.get_variable("W1", initializer=xavier_init, shape=[self.config.n_features* self.config.embed_size, self.config.hidden_size])
        b1 = tf.get_variable("b1", initializer=xavier_init, shape=[self.config.hidden_size,])
        W2 = tf.get_variable("W2", initializer=xavier_init, shape=[self.config.n_features* self.config.embed_size, self.config.hidden_size])
        b = tf.get_variable("b2", initializer=xavier_init,  shape=[1,])

        # relu, dropout
        h1 = tf.nn.relu(tf.matmul(z1, W1) + tf.matmul(x1, W2) + b1)
        h2 = tf.nn.relu(tf.matmul(z2, W1) + tf.matmul(x2, W2) + b1)

        if self.config.second_hidden_size is not None:
            U = tf.get_variable("U", shape = (1, self.config.second_hidden_size), initializer=xavier_init, dtype=tf.float32)
            W3 = tf.get_variable("W3", initializer=xavier_init, shape=[self.config.hidden_size, self.config.second_hidden_size])
            b3 = tf.get_variable("b3", initializer=xavier_init, shape=[self.config.second_hidden_size,])
            r1 = tf.nn.relu(tf.matmul(h1, W3) + b3)
            r2 = tf.nn.relu(tf.matmul(h2, W3) + b3)
            # r1_drop = tf.nn.dropout(r1, keep_prob=dropout_rate)
            # r2_drop = tf.nn.dropout(r2, keep_prob=dropout_rate)
            preds = tf.reduce_sum(U * tf.sub(r1, r2), 1) + b

        else:
            U = tf.get_variable("U", shape=(1, self.config.hidden_size), initializer=xavier_init, dtype=tf.float32)
            if self.config.add_distance:
                # U2 = tf.get_variable("U2", shape=(1, self.config.hidden_size), initializer=xavier_init, dtype=tf.float32)
                a = tf.get_variable("a", initializer=xavier_init, shape=[1,])
                diff_12 = tf.nn.dropout(h1 - h2, keep_prob=dropout_rate)
                sqdiff_12 = tf.square(diff_12)
                sqdist_12 = tf.reduce_sum(sqdiff_12, 1)
                inner_12 = tf.reduce_sum(U * h1 * h2, 1)
                # inner_dist_12 = tf.reduce_sum(U2 * h1_drop * h2_drop, 1)
                preds = inner_12 +  a * sqdist_12 + b
            else:
                preds = tf.reduce_sum(U * h1 * h2, 1) + b

        return preds

    def add_exact_prediction_op(self, preds):
        pos_thres = tf.constant(0.5, dtype=tf.float32, shape=(1,))
        return tf.greater(tf.sigmoid(preds), pos_thres)

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        Compute averaged cross entropy loss for the predictions.
        Importantly, you must ignore the loss for any masked tokens.

        Args:
            pred: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.labels_placeholder, logits=preds, pos_weight=self.config.pos_weight))
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.config.lr
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   200000, self.config.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def preprocess_sequence_data(self, examples):
        return pad_sequences(examples, self.max_length)

def pad_sequences(data, max_length, n_features=1):
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * n_features


    for sentence1, sentence2, label in data:
        feat_sent1 = zero_vector * max_length
        feat_sent2 = zero_vector * max_length
        feat_mask1 = [0.0] * max_length
        feat_mask2 = [0.0] * max_length
        for i, word in enumerate(sentence1):
            if i >= max_length:
                break
            feat_sent1[i] = word
            feat_mask1[i] = 1.0
        for i, word in enumerate(sentence2):
            if i >= max_length:
                break
            feat_sent2[i] = word
            feat_mask2[i] = 1.0
        seqlen1 = min(len(sentence1), max_length)
        seqlen2 = min(len(sentence2), max_length)
        ret.append((feat_sent1, feat_sent2, seqlen1, seqlen2, feat_mask1, feat_mask2, label))
    return ret
