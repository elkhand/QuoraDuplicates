# QuoraDuplicates
CS224N Final Project: Detecting if 2 questions are duplicates of each other.

## How to run the model for Entailment paper

From the project's root directory (assuming you created data directory in <project-root> dir):

Running Attention model:

```
python run.py train \
-m attention \
-dt1 data/dat_train_a.conll \
-dt2 data/dat_train_b.conll \
-dtl data/labels_train.conll \
-dd1 data/dat_dev_a.conll \
-dd2 data/dat_dev_b.conll \
-ddl data/labels_dev.conll \
-de1 data/dat_test_a.conll \
-de2 data/dat_test_b.conll \
-v data/glvocab_1.txt \
-vv data/glwordvectors_1_100.txt \
-eb 100 \
-cfg config/attention_0.py
```

Running Simaese model:

```
python run.py train \
-m siamese \
-dt1 data/dat_train_a.conll \
-dt2 data/dat_train_b.conll \
-dtl data/labels_train.conll \
-dd1 data/dat_dev_a.conll \
-dd2 data/dat_dev_b.conll \
-ddl data/labels_dev.conll \
-de1 data/dat_test_a.conll \
-de2 data/dat_test_b.conll \
-v data/glvocab_1.txt \
-vv data/glwordvectors_1_300.txt \
-eb 300 \
-cfg config/siamese_0.py
```

Running sendEmail.py script:

```
python sendEmail.py elkhan.dadashov@gmail.com /home/elkhand/QuoraDuplicates/exp_march9_relu_pred.txt

python sendEmail.py <your_email> <path_to_file_to_be_attached_to_email>
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
