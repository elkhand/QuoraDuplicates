# QuoraDuplicates
CS224N Final Project: Detecting if 2 questions are duplicates of each other.

## How to run the model for Entailment paper

From the project's root directory (assuming you created data directory in <project-root> dir):

Running Attention-Entailment model:

```
python attention-entailment-model/run.py train \
-dt1 data/dat_train_a.conll \
-dt2 data/dat_train_b.conll \
-dtl data/labels_train.conll \
-dd1 data/dat_dev_a.conll \
-dd2 data/dat_dev_b.conll \
-ddl data/labels_dev.conll \
-v data/glvocab_1_100.txt \
-vv data/glwordvectors_1_100.txt
```

Running Simaese model:

```
python siamese-model/run.py train \
-dt1 data/dat_train_a.conll \
-dt2 data/dat_train_b.conll \
-dtl data/labels_train.conll \
-dd1 data/dat_dev_a.conll \
-dd2 data/dat_dev_b.conll \
-ddl data/labels_dev.conll \
-v data/glvocab_1_100.txt \
-vv data/glwordvectors_1_100.txt
```


## You can download data from this link:

https://drive.google.com/open?id=0B9-tEMxgDDsVRlVGb0ZkUHB2aFE 


## Proposal Google Doc:

https://docs.google.com/document/d/1b9ItoYLbtbE02FOlK1V5eEH_6HvlvUDF_X-QHQPzDoo/edit?usp=sharing 

## Quora Dataset Details:

https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs

## Related Work

- Reasoning about Entailment with Neural Attention https://arxiv.org/pdf/1509.06664v1.pdf
- Siamese Recurrent Architectures for Learning Sentence Similarity http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf 
- Efficient mining of hard examples for object detection http://www.idiap.ch/~fleuret/SMLD/2014/SMLD2014_-_Olivier_Canevet_-_Efficient_mining_of_hard_examples_for_object_detection.pdf 
