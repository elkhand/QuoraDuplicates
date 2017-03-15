import os
from datetime import datetime

class Config:
    n_word_features = 1 # Number of features for every word in the input.
    n_features = n_word_features # Number of features for every word in the input.
    max_length = 35 # longest sequence to parse
    n_classes = 2
    dropout = 0.7
    hidden_size = 700
    second_hidden_size=None
    batch_size = 100
    n_epochs = 1000
    lr = 0.0003#0.0003 F1: 76 Acc: 83
    lr_decay_rate = 0.9
    embeddings_trainable = False
    pos_weight = 1.7
    # bidirectional = False
    add_distance = False
    beta = 0.005
    

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
