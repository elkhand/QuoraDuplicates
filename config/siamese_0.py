import os
from datetime import datetime

class Config:
    n_word_features = 1 # Number of features for every word in the input.
    n_features = n_word_features # Number of features for every word in the input.
    max_length = 20 # longest sequence to parse
    n_classes = 2
    dropout = 0.5
    hidden_size = 1100
    second_hidden_size = 800
    batch_size = 100
    n_epochs = 1000
    lr = 0.0006#0.0003 F1: 76 Acc: 83
    lr_decay_rate = 0.1
    embeddings_trainable = True
    pos_weight = 1.7
    beta = 0.1
    bidirectional = False

    def __init__(self, args):
        self.cell = "lstm"

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.dev_prob_output = self.output_path + "dev_pred_probs.txt"
        self.isEnsembleOn = False
        if self.isEnsembleOn:
            self.attention_dev_prob_output = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/att_test_dev_pred_probs.txt"
        
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)

        self.log_output = self.output_path + "log"
        self.embed_size = int(args.embed_size)
