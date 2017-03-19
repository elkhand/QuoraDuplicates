
from __future__ import absolute_import
from __future__ import division
import logging

import tensorflow as tf
import numpy as np

from util import Progbar, minibatches
from model import Model


logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class SiameseModel(Model):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    """

    def add_placeholders(self):
        self.input1_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.input2_placeholder = tf.placeholder(tf.int32, (None, self.max_length))
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None,2))
        self.seqlen1_placeholder = tf.placeholder(tf.int32, (None,))
        self.seqlen2_placeholder = tf.placeholder(tf.int32, (None,))
        self.dropout_placeholder = tf.placeholder(tf.float32, [])

    def create_feed_dict(self, inputs1_batch, inputs2_batch, seqlen1_batch, seqlen2_batch, labels_batch=None, dropout=1):
        feed_dict = {
            self.input1_placeholder: inputs1_batch,
            self.input2_placeholder: inputs2_batch,
            self.seqlen1_placeholder : seqlen1_batch,
            self.seqlen2_placeholder : seqlen2_batch,
            self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            sp_labels = []
            for l in labels_batch:
                if l==0:
                    sp_labels.append([1,0])
                else:
                    sp_labels.append([0,1])
            sp_labels= np.array(sp_labels,dtype=np.float32)
            feed_dict.update({self.labels_placeholder: sp_labels})
        return feed_dict

    def add_embedding(self, ind):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, n_features * embed_size).

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        if self.config.embeddings_trainable:
            embeddings = tf.Variable(self.pretrained_embeddings, name="embeddings")
        else:
            embeddings = self.pretrained_embeddings

        if ind==1:
            to_concat = tf.nn.embedding_lookup(embeddings, self.input1_placeholder)
        if ind==2:
            to_concat = tf.nn.embedding_lookup(embeddings, self.input2_placeholder)
        embeddings = tf.reshape(to_concat, [-1, self.config.max_length, self.config.n_features* self.config.embed_size])
        return embeddings

    def add_prediction_op(self):
        """Adds the unrolled RNN:

        Remember:
            * Use the xavier initilization for matrices.
            * Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """

        x1 = self.add_embedding(1)
        x2 = self.add_embedding(2)
        dropout_rate = self.dropout_placeholder

        BasicLSTMCell = tf.contrib.rnn.BasicLSTMCell if hasattr(tf.contrib.rnn, 'BasicLSTMCell') else tf.nn.rnn_cell.BasicLSTMCell
        LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple if hasattr(tf.contrib.rnn, 'LSTMStateTuple') else tf.nn.rnn_cell.LSTMStateTuple
        # DropoutWrapper = tf.contrib.rnn.DropoutWrapper if hasattr(tf.contrib.rnn, 'DropoutWrapper') else tf.nn.rnn_cell.DropoutWrapper

        if self.config.cell == "lstm":
            cell = BasicLSTMCell(self.config.hidden_size)
            # cell = DropoutWrapper(cell, dropout_rate)
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)

        xavier_init = tf.contrib.layers.xavier_initializer()


        m = self.config.second_hidden_size
        with tf.variable_scope("HiddenLayerVars"):
            b1 = tf.get_variable("b1",initializer=xavier_init,  shape=[1, m])
            b2 = tf.get_variable("b2",initializer=xavier_init,  shape=[1,2])
            W1 = tf.get_variable("W1",initializer=xavier_init, shape=[4*(self.config.hidden_size+1)+1, m])
            W2 = tf.get_variable("W2",initializer=xavier_init, shape=[m, 2])
            tf.get_variable_scope().reuse_variables()

        # Initialize state as vector of zeros.
        batch_size = tf.shape(x1)[0]
        h1 = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        c1 = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        h2 = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        c2 = tf.zeros([batch_size, self.config.hidden_size], dtype=tf.float32)
        with tf.variable_scope("LSTM"):
            Y, (c1, h1) = tf.nn.dynamic_rnn(cell, x1, initial_state=LSTMStateTuple(c1, h1), sequence_length=self.seqlen1_placeholder)
            #h1_drop = tf.nn.dropout(h1, keep_prob=dropout_rate)
            tf.get_variable_scope().reuse_variables()
            Y, (c2, h2) = tf.nn.dynamic_rnn(cell, x2, initial_state=LSTMStateTuple(c2, h2), sequence_length=self.seqlen2_placeholder)
            #h2_drop = tf.nn.dropout(h2, keep_prob=dropout_rate)

        # add sentence length
        sent_len1 = tf.cast(tf.reshape(self.seqlen1_placeholder, [batch_size,1]), tf.float32) / float(self.config.max_length)
        sent_len2 = tf.cast(tf.reshape(self.seqlen2_placeholder, [batch_size,1]), tf.float32) / float(self.config.max_length)
        if int(tf.__version__.split('.')[0]) >= 1: # TensorFlow 1.0 or greater
            h1 = tf.concat([h1, sent_len1], 1) # hidden_size+1
            h2 = tf.concat([h2, sent_len2], 1) # hidden_size+1
        else:
            h1 = tf.concat(1, [h1, sent_len1]) # hidden_size+1
            h2 = tf.concat(1, [h2, sent_len2]) # hidden_size+1

        h_sub = tf.subtract(h1, h2)
        sqdiff_12 = tf.square(h_sub)
        sqdist_12 = tf.reduce_sum(sqdiff_12, 1)
        h_dist = tf.reshape(sqdist_12, [batch_size,1])
        h_mul =  tf.multiply(h1 , h2)

        if int(tf.__version__.split('.')[0]) >= 1: # TensorFlow 1.0 or greater
            h_combined = tf.concat([h1, h2, h_sub, h_dist, h_mul], 1) # 4*(hidden_size+1)+1
        else:
            h_combined = tf.concat(1, [h1, h2, h_sub, h_dist, h_mul]) # 4*(hidden_size+1)+1
        h_combined_drop = tf.nn.dropout(h_combined, keep_prob=dropout_rate)

        e1 = tf.matmul(h_combined_drop, W1) + b1 # [bath_size,m]
        e1_relu = tf.nn.relu(e1)
        e1_drop = tf.nn.dropout(e1_relu, keep_prob=dropout_rate)
        preds = tf.matmul(e1_drop,W2) + b2

        return preds

    def add_exact_prediction_op(self, preds):
        pos_thres = tf.constant(0.5, dtype=tf.float32, shape=(1,))
        return tf.greater(tf.sigmoid(preds[:, 1]), pos_thres)

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

        m = self.config.second_hidden_size
        xavier_init = tf.contrib.layers.xavier_initializer()
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.labels_placeholder, logits=preds, pos_weight=1.675))
        with tf.variable_scope("HiddenLayerVars", reuse=True):
            b1 = tf.get_variable("b1",initializer=xavier_init,  shape=[1, m])
            b2 = tf.get_variable("b2",initializer=xavier_init,  shape=[1, 2])
            W1 = tf.get_variable("W1",initializer=xavier_init, shape=[4*(self.config.hidden_size+1)+1, m])
            W2 = tf.get_variable("W2",initializer=xavier_init, shape=[m, 2])
            tf.get_variable_scope().reuse_variables()
        loss = loss + self.config.beta*tf.nn.l2_loss(W1)+ self.config.beta*tf.nn.l2_loss(W2)

        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        global_step = tf.Variable(0, name='global_step', trainable=False)
        starter_learning_rate = self.config.lr
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   85015, self.config.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
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



# INFO:Epoch 1 out of 1000
#    1/2429 [..............................] - ETA: 4363s - train loss: 68.8533I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 3015 get requests, put_count=2235 evicted_count=1000 eviction_rate=0.447427 and unsatisfied allocation rate=0.623549
#   13/2429 [..............................] - ETA: 748s - train loss: 50.3973I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 1656 get requests, put_count=1941 evicted_count=1000 eviction_rate=0.515198 and unsatisfied allocation rate=0.445652
#   33/2429 [..............................] - ETA: 538s - train loss: 31.1680I tensorflow/core/common_runtime/gpu/pool_allocator.cc:247] PoolAllocator: After 4780 get requests, put_count=4825 evicted_count=1000 eviction_rate=0.207254 and unsatisfied allocation rate=0.212134
# 2429/2429 [==============================] - 410s - train loss: 1.0789      e_limit_ from 655 to 720
#
# INFO:Evaluating on training data: 10k sample
# 100/101 [============================>.] - ETA: 0sCorrect_preds:  3194  Total_preds:  4220.0    Total_correct:  3773.0
# INFO:acc/P/R/F1/loss: 0.840/0.757/0.847/0.799/0.5041
# INFO:Evaluating on development data
# 809/809 [==============================] - 28s
# Correct_preds:  24154   Total_preds:  34413.0   Total_correct:  29757.0
# INFO:acc/P/R/F1/loss: 0.804/0.702/0.812/0.753/0.5621
# INFO:New best score! Saving model in /home/elkhand/QuoraDuplicates-2/results/lstm/20170316_010529/model.weights
#
# INFO:Epoch 2 out of 1000
# 2429/2429 [==============================] - 411s - train loss: 0.4685
#
# INFO:Evaluating on training data: 10k sample
# 100/101 [============================>.] - ETA: 0sCorrect_preds:  3554  Total_preds:  4288.0    Total_correct:  3773.0
# INFO:acc/P/R/F1/loss: 0.905/0.829/0.942/0.882/0.3544
# INFO:Evaluating on development data
# 809/809 [==============================] - 28s
# Correct_preds:  25279   Total_preds:  35336.0   Total_correct:  29757.0
# INFO:acc/P/R/F1/loss: 0.820/0.715/0.850/0.777/0.5137
# INFO:New best score! Saving model in /home/elkhand/QuoraDuplicates-2/results/lstm/20170316_010529/model.weights
#
# INFO:Epoch 3 out of 1000
#  364/2429 [===>..........................] - ETA: 355s - train loss: 0.3208

# beta=0.1, max_len=20
# ~/QuoraDuplicates-2/results/lstm/20170316_012847/results.txt
#83.4 78.2

# beta=0.12, max_len=20
#/home/ubuntu/rnn/results/lstm/20170316_070844/model.weights
#83.4 78.3

# beta=0.15, max_len=20
#/home/elkhand/QuoraDuplicates-2/results/lstm/20170316_055337/model.weights
#83.1 78.5

# beta=0.05, max_len=20


# beta=0.08, max_len=40




# beta=0.1, max_len=40
#/home/ubuntu/rnn/results/lstm/20170316_031118/model.weights
#83.3 79.0

# beta=0.15, max_len=40
#83.1 78.3

# beta=0.2, max_len=40
# rnn/results/lstm/20170316_082325/
# 83.9 78.2
# 83.3 78.3
# 84.0 77.4

# beta=0.25, max_len=40
#~/rnn/results/lstm/20170316_082325/results.txt
#83.3 78.5
