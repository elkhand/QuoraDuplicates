import os
from datetime import datetime


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_word_features = 1 # Number of features for every word in the input.
    n_features = n_word_features # Number of features for every word in the input.
    max_length = 40 # longest sequence to parse
    n_classes = 2
    dropout = 0.99
    hidden_size = 512
    second_hidden_size = None
    batch_size = 100
    n_epochs = 100
    max_grad_norm = 10.
    lr = 0.0003
    lr_decay_rate = 0.9
    embeddings_trainable = False
    pos_weight = 1.7
    add_distance = True

    def __init__(self, args):
        self.cell = "lstm"

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)

        self.log_output = self.output_path + "log"
        self.embed_size = int(args.embed_size)
