import logging

import tensorflow as tf
import numpy as np

from util import Progbar, minibatches

logger = logging.getLogger("fp")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """

    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph."""
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict.

        Note: The signature of this function must match the return value of preprocess_sequence_data.

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_exact_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into exact predictions.

        Each prediction must be either 0 or 1.

        Returns:
            pred: A tensor of shape (batch_size,)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def preprocess_sequence_data(self, examples):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_embedding(self, ind):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        if self.config.embeddings_trainable:
            embeddings = tf.Variable(self.pretrained_embeddings, name="embeddings")
        else:
            embeddings = self.pretrained_embeddings

        if ind == 1:
            to_concat = tf.nn.embedding_lookup(embeddings, self.input1_placeholder)
        elif ind == 2:
            to_concat = tf.nn.embedding_lookup(embeddings, self.input2_placeholder)
        embeddings = tf.reshape(to_concat, [-1, self.config.max_length, self.config.n_features * self.config.embed_size])
        return embeddings

    def evaluate(self, sess, inputs_raw):
        """Evaluates model performance on @examples."""
        inputs = self.preprocess_sequence_data(inputs_raw)
        labels = [label for sentence1, sentence2, label in inputs_raw]
        return self._evaluate(sess, inputs, labels, isDev=True)

    def _evaluate(self, sess, inputs, labels, isDev=False):
        preds, logits, loss, _ = self._output(sess, inputs)
        labels = np.array(labels, dtype=np.float32)
        preds = np.array(preds)
        probs = logits
        if isDev:
            # store dev prediction probabilities
            prob_predSM = softmax(np.array(probs))
            with open(self.config.dev_prob_output, 'a') as f:
                np.savetxt(f, prob_predSM, fmt='%1.10f', delimiter=' ', newline='\n')

        #---Ensemble part
        if self.config.isEnsembleOn and isDev:
            otherModelProbs=[]
            thisProbs = np.array(probs)
            thisProbs = softmax(thisProbs)
            with open(self.config.attention_dev_prob_output, 'r') as f:
                otherModelProbs = np.loadtxt(f)
            otherModelProbs = otherModelProbs[self.epochNum*len(labels):(self.epochNum+1)*len(labels)]
            sumOfProbs = np.add(thisProbs, otherModelProbs)
            avgOfProbs = np.divide(sumOfProbs,2.0)
            newPreds = [0 if diff > same else 1 for diff,same in avgOfProbs]
            preds = np.asarray(newPreds)
        #End of Ensemble part


        correct_preds = np.logical_and(labels==1, preds==1).sum()
        total_preds = float(np.sum(preds==1))
        total_correct = float(np.sum(labels==1))

        print "Correct_preds: ",correct_preds,"\tTotal_preds: ", total_preds,"\tTotal_correct: ", total_correct

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = sum(labels==preds) / float(len(labels))
        return (acc, p, r, f1, loss, logits, labels, preds)

    def output(self, sess, inputs_raw, extra_fetch=[]):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        inputs = self.preprocess_sequence_data(inputs_raw)
        return self._output(sess, inputs, extra_fetch)

    def _output(self, sess, inputs, extra_fetch=[]):
        preds = []
        logits = []
        loss_record = []
        extras = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # batch = batch[:4] # ignore label
            feed = self.create_feed_dict(*batch)
            preds_, logits_, loss_, extra_ = sess.run([self.predictions, self.pred, self.loss, extra_fetch], feed_dict=feed)
            preds += list(preds_)
            loss_record.append(loss_)
            logits += list(logits_)
            if extra_fetch:
                extras.append(extra_)
            prog.update(i + 1, [])
        if extra_fetch:
            extras = np.concatenate(extras)
        return preds, logits, np.mean(loss_record), extras

    def _run_epoch(self, sess, train, train_labels, dev, dev_labels):
        prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        for i, batch in enumerate(minibatches(train, self.config.batch_size)):
            feed = self.create_feed_dict(*batch, dropout=self.config.dropout)
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
            prog.update(i + 1, [("train loss", loss)])

            if self.report: self.report.log_train_loss(loss)
        print("")

        logger.info("Evaluating on training data: 10k sample")
        n_train_evaluate = 10000
        train_entity_scores = self._evaluate(sess, train[:n_train_evaluate], train_labels[:n_train_evaluate])
        train_entity_scores = train_entity_scores[:5]
        logger.info("acc/P/R/F1/loss: %.3f/%.3f/%.3f/%.3f/%.4f", *train_entity_scores)

        logger.info("Evaluating on development data")
        entity_scores = self._evaluate(sess, dev, dev_labels, isDev=True)
        entity_scores = entity_scores[:5]
        logger.info("acc/P/R/F1/loss: %.3f/%.3f/%.3f/%.3f/%.4f", *entity_scores)

        with open(self.config.eval_output, 'a') as f:
            f.write('%.4f %.4f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n' % (train_entity_scores[4], entity_scores[4], train_entity_scores[0], entity_scores[0], train_entity_scores[3], entity_scores[3], entity_scores[0], entity_scores[1], entity_scores[2]))

        f1 = entity_scores[-2]
        return f1

    def fit(self, sess, saver, train_raw, dev_raw):
        best_score = 0.

        # Padded sentences
        train = self.preprocess_sequence_data(train_raw)
        train_labels = [label for sentence1, sentence2, label in train_raw]
        dev = self.preprocess_sequence_data(dev_raw)
        dev_labels = [label for sentence1, sentence2, label in dev_raw]

        for epoch in range(self.config.n_epochs):
            self.epochNum = epoch
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self._run_epoch(sess, train, train_labels, dev, dev_labels)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
        return best_score

    def _build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.predictions = self.add_exact_prediction_op(self.pred)

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        self.helper = helper
        self.config = config
        self.report = report
        self.epochNum = 0
        self.max_length = min(self.config.max_length, helper.max_length)
        self.config.max_length = self.max_length # Just in case people make a mistake.
        self.pretrained_embeddings = pretrained_embeddings

        self._build()

def softmax(x):
    if len(x.shape) > 1:
        t = np.max(x, axis = 1)
        x -= t.reshape((x.shape[0], 1))
        x = np.exp(x)
        t = np.sum(x, axis = 1)
        x /= t.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        t = np.sum(x)
        x /= t
    return x
