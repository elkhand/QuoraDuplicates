from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

from model import Model

class AttentionModel(Model):
    """
    Implements a sequence-to-sequence network with word-by-word attention.
    """
    def add_placeholders(self):
        self.input1_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.input2_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.seqlen1_placeholder = tf.placeholder(tf.int32, (None,))
        self.seqlen2_placeholder = tf.placeholder(tf.int32, (None,))
        self.mask1_placeholder = tf.placeholder(tf.float32, (None, self.max_length))
        self.mask2_placeholder = tf.placeholder(tf.float32, (None, self.max_length))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.dropout_placeholder = tf.placeholder(tf.float32, [])

    def create_feed_dict(self, inputs1_batch, inputs2_batch, seqlen1_batch, seqlen2_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict.

        Note: The signature of this function must match the return value of preprocess_sequence_data.

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        max_length = self.config.max_length
        inf = 10000.0  # Use a pseudo infinity for numerical stability.
        mask1_batch = [([0.0] * seqlen) + ([-inf] * (max_length - seqlen)) for seqlen in seqlen1_batch]
        mask2_batch = [([0.0] * seqlen) + ([-inf] * (max_length - seqlen)) for seqlen in seqlen2_batch]

        feed_dict = {
            self.input1_placeholder: inputs1_batch,
            self.input2_placeholder: inputs2_batch,
            self.seqlen1_placeholder: seqlen1_batch,
            self.seqlen2_placeholder: seqlen2_batch,
            self.mask1_placeholder: mask1_batch,
            self.mask2_placeholder: mask2_batch,
            self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            feed_dict.update({self.labels_placeholder: np.array(labels_batch, dtype=np.float32)})
        return feed_dict

    def _add_asymmetric_prediction_op(self, x1, x2, seqlen1, seqlen2, mask1):
        batch_size = tf.shape(x1)[0]
        hidden_size = self.config.hidden_size
        # h_step1, h_step2 = list(), list()

        # TensorFlow 1.0 compatibility
        BasicLSTMCell = tf.contrib.rnn.BasicLSTMCell if hasattr(tf.contrib.rnn, 'BasicLSTMCell') else tf.nn.rnn_cell.BasicLSTMCell
        LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple if hasattr(tf.contrib.rnn, 'LSTMStateTuple') else tf.nn.rnn_cell.LSTMStateTuple
        DropoutWrapper = tf.contrib.rnn.DropoutWrapper if hasattr(tf.contrib.rnn, 'DropoutWrapper') else tf.nn.rnn_cell.DropoutWrapper
        def concat(values, axis):
            if int(tf.__version__.split('.')[0]) >= 1: # TensorFlow 1.0 or greater
                return tf.concat(values, axis)
            return tf.concat(axis, values)

        if self.config.cell == "lstm":
            cell1 = BasicLSTMCell(self.config.hidden_size)
            cell2 = BasicLSTMCell(self.config.hidden_size)

            cell1 = DropoutWrapper(cell1, self.dropout_placeholder)
            cell2 = DropoutWrapper(cell2, self.dropout_placeholder)
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)

        # Initialize state as vector of zeros.
        c = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        h = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)

        with tf.variable_scope("LSTM1"):
            Y, (c, h) = tf.nn.dynamic_rnn(cell1, x1, initial_state=LSTMStateTuple(c, h), sequence_length=seqlen1)
            # for time_step in range(self.max_length):
            #     x_t = x1[:, time_step, :]
            #     _, h, c = cell1(x_t, h, c)
            #     h_step1.append(h)
            #     if time_step == 0:
            #         tf.get_variable_scope().reuse_variables()

        # Use c from the output of the 1st LSTM as input to the 2nd, and reset h.
        h = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)

        with tf.variable_scope("LSTM2"):
            Y2, (c, h) = tf.nn.dynamic_rnn(cell2, x2, initial_state=LSTMStateTuple(c, h), sequence_length=seqlen2)
            # for time_step in range(self.max_length):
            #     x_t = x2[:, time_step, :]
            #     _, h, c = cell2(x_t, h, c)
            #     h_step2.append(h)
            #     if time_step == 0:
            #         tf.get_variable_scope().reuse_variables()
            last_h = h

        # setup Y if adding embeddings
        if self.config.score_type == 3:
            Y = concat([Y, x1], 2)  # (?, L, hidden_size+embed_size)

        if self.config.uses_attention:
            xavier_init = tf.contrib.layers.xavier_initializer()
            W_y = tf.get_variable("W_y", shape=[self.config.hidden_size, self.config.hidden_size], initializer=xavier_init)
            if self.config.score_type == 3:
                W_h = tf.get_variable("W_h", shape=[self.config.hidden_size + self.config.embed_size, self.config.hidden_size + self.config.embed_size], initializer=xavier_init)
                W_r = tf.get_variable("W_r", shape=[self.config.hidden_size + self.config.embed_size, self.config.hidden_size + self.config.embed_size], initializer=xavier_init)
                W_t = tf.get_variable("W_t", shape=[self.config.hidden_size + self.config.embed_size, self.config.hidden_size + self.config.embed_size], initializer=xavier_init)
                W_p = tf.get_variable("W_p", shape=[self.config.hidden_size + self.config.embed_size, self.config.hidden_size], initializer=xavier_init)
                W_x = tf.get_variable("W_x", shape=[self.config.hidden_size, self.config.hidden_size], initializer=xavier_init)
            else:
                W_h = tf.get_variable("W_h", shape=[self.config.hidden_size, self.config.hidden_size], initializer=xavier_init)
                W_r = tf.get_variable("W_r", shape=[self.config.hidden_size, self.config.hidden_size], initializer=xavier_init)
                w = tf.get_variable("w", shape=[self.config.hidden_size], initializer=xavier_init)
                W_t = tf.get_variable("W_t", shape=[self.config.hidden_size, self.config.hidden_size], initializer=xavier_init)
                W_p = tf.get_variable("W_p", shape=[self.config.hidden_size, self.config.hidden_size], initializer=xavier_init)
                W_x = tf.get_variable("W_x", shape=[self.config.hidden_size, self.config.hidden_size], initializer=xavier_init)

            # Precompute W_y * Y, because it's used many times in the loop.
            # Y's shape is (?, L, hidden_size)
            # W_y's shape is (hidden_size, hidden_size)
            tmp = tf.reshape(Y, [-1, hidden_size])  # (? * L, hidden_size)
            tmp2 = tf.matmul(tmp, W_y)  # (? * L, hidden_size)
            W_y_Y = tf.reshape(tmp2, [-1, self.max_length, hidden_size])  # (?, L, hidden_size)

            # Initialize r_0 to zeros.
            if self.config.score_type == 3:
                r_t = tf.zeros([batch_size, self.config.hidden_size+self.config.embed_size], dtype=tf.float32)
            else:
                r_t = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
            r_step = []
            alpha_step = []

            for time_step in range(self.max_length):
                h_t = Y2[:, time_step, :]

                if self.config.score_type == 3:
                    # M_t = Y .* ((W_h * [h_t, x_t]) + (W_r * r_{t-1})) X e_L)
                    x_t = x2[:, time_step, :]
                    input = concat([h_t, x_t], 1)  # (?, hidden_size+embed_size)
                    tmp = tf.matmul(input, W_h) + tf.matmul(r_t, W_r)  # (?, hidden_size+embed_size)
                    tmp2 = tf.tile(tf.expand_dims(tmp, 1), (1, self.max_length, 1))  # (?, L, hidden_size+embed_size)
                    M_t = Y * tmp2  # (?, L, hidden_size)
                    alpha_t = tf.nn.softmax(tf.reduce_sum(M_t, 2) + mask1)

                    # r_t = (Y * alpha_t^T) + tanh(W_t * r_{t-1})
                    tmp3 = tf.tile(tf.expand_dims(alpha_t, 2), (1, 1, hidden_size+self.config.embed_size))  # (?, L, hidden_size+embed_size)
                    Y_alpha_t = tf.reduce_sum(Y * tmp3, 1)  # (?, hidden_size+embed_size)
                    r_t = Y_alpha_t + tf.tanh(tf.matmul(r_t, W_t))  # (?, hidden_size+embed_size)

                elif self.config.score_type == 2:
                    # M_t = Y ^T ((W_h * h_t) + (W_r * r_{t-1})) X e_L)
                    tmp = tf.matmul(h_t, W_h) + tf.matmul(r_t, W_r)  # (?, hidden_size)
                    tmp2 = tf.tile(tf.expand_dims(tmp, 1), (1, self.max_length, 1))  # (?, L, hidden_size)
                    M_t = Y * tmp2  # (?, L, hidden_size)
                    alpha_t = tf.nn.softmax(tf.reduce_sum(M_t, 2) + mask1)

                    # r_t = (Y * alpha_t^T) + tanh(W_t * r_{t-1})
                    tmp3 = tf.tile(tf.expand_dims(alpha_t, 2), (1, 1, hidden_size))  # (?, L, hidden_size)
                    Y_alpha_t = tf.reduce_sum(Y * tmp3, 1)  # (?, hidden_size)
                    r_t = Y_alpha_t + tf.tanh(tf.matmul(r_t, W_t))  # (?, hidden_size)

                else:
                    # M_t = tanh((W_y * Y) + ((W_h * h_t) + (W_r * r_{t-1})) X e_L)
                    tmp = tf.matmul(h_t, W_h) + tf.matmul(r_t, W_r)  # (?, hidden_size)
                    tmp2 = tf.tile(tf.expand_dims(tmp, 1), (1, self.max_length, 1))  # (?, L, hidden_size)
                    M_t = tf.tanh(W_y_Y + tmp2)  # (?, L, hidden_size)

                    # alpha_t = softmax(w^T * M_t)
                    alpha_t = tf.nn.softmax(tf.reduce_sum(M_t * w, 2) + mask1)  # (?, L)

                    # r_t = (Y * alpha_t^T) + tanh(W_t * r_{t-1})
                    tmp3 = tf.tile(tf.expand_dims(alpha_t, 2), (1, 1, hidden_size))  # (?, L, hidden_size)
                    Y_alpha_t = tf.reduce_sum(Y * tmp3, 1)  # (?, hidden_size)
                    r_t = Y_alpha_t + tf.tanh(tf.matmul(r_t, W_t))  # (?, hidden_size)

                r_step.append(r_t)
                alpha_step.append(alpha_t)

            # h* = tanh((W_p * r_L) + (W_x * h_N))
            r = tf.transpose(tf.stack(r_step), [1, 2, 0])  # (?, hidden_size, L)
            tmp4 = tf.one_hot(seqlen2 - 1, self.max_length, dtype=tf.float32)  # (?, L)
            r_L = tf.squeeze(tf.matmul(r, tf.expand_dims(tmp4, 2)), 2)  # (?, hidden_size)
            last_h = tf.tanh(tf.matmul(r_L, W_p) + tf.matmul(last_h, W_x))  # (?, hidden_size)

            # Create a node for alpha so that it can be inspected by the shell to create
            # the attention visualization.
            alpha = tf.transpose(tf.stack(alpha_step), [1, 0, 2], "alpha")  # (?, L, L)

        return last_h

    def add_prediction_op(self):
        x1 = self.add_embedding(1)
        x2 = self.add_embedding(2)

        with tf.variable_scope("LSTM_attention"):
            # Define U and b as variables.
            xavier_init = tf.contrib.layers.xavier_initializer()
            U = tf.get_variable("U", shape=(self.config.hidden_size, 2), dtype=tf.float32, initializer=xavier_init)
            b = tf.Variable(initial_value=np.zeros((1, 2)), dtype=tf.float32, name="b")

            last_h_a = self._add_asymmetric_prediction_op(x1, x2, self.seqlen1_placeholder, self.seqlen2_placeholder, self.mask1_placeholder)
            tf.get_variable_scope().reuse_variables()
            last_h_b = self._add_asymmetric_prediction_op(x2, x1, self.seqlen2_placeholder, self.seqlen1_placeholder, self.mask2_placeholder)

            last_h = last_h_a + last_h_b

            # use U and b for final prediction
            h_drop = tf.nn.dropout(last_h, keep_prob=self.dropout_placeholder)
            preds = tf.matmul(h_drop, U) + b

        return preds

    def add_exact_prediction_op(self, preds):
        return tf.argmax(preds, 1)

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
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=preds)
        loss = tf.reduce_mean(loss)
        with tf.variable_scope("LSTM_attention", reuse=True):
            U = tf.get_variable("U")
        return loss + 0.1 * tf.nn.l2_loss(U)

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        global_step = tf.Variable(0, trainable=False, name="global_step")
        starter_learning_rate = self.config.lr
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   200000, self.config.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, grad_vars = zip(*grads_and_vars)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.config.max_grad_norm)
        train_op = optimizer.apply_gradients(zip(grads, grad_vars))
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
        for i, word in enumerate(sentence1):
            if i >= max_length:
                break
            feat_sent1[i] = word

        for i, word in enumerate(sentence2):
            if i >= max_length:
                break
            feat_sent2[i] = word
        seqlen1 = min(len(sentence1), max_length)
        seqlen2 = min(len(sentence2), max_length)
        ret.append((feat_sent1, feat_sent2, seqlen1, seqlen2, label))
    return ret
