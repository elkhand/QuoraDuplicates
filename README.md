# QuoraDuplicates
CS224N Final Project: Determining whether two questions are duplicates of each other.

## How to run the models

1. Create a folder named "data"
2. Download the Quora dataset from http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv
3. Use scripts/TokenizeQuestions-WordVectors.ipynb to generate the training/dev/test sets and put them inside the data folder. Alternatively, you can also download the data from https://drive.google.com/open?id=0B9-tEMxgDDsVRlVGb0ZkUHB2aFE
4. Download the GloVe data from http://nlp.stanford.edu/data/glove.6B.zip
5. Use scripts/gen_data.py to generate glvocab_1.txt and glwordvectors_1_[100/300].txt and put them inside the data folder.
6. Depending on what you want to do, use one of the following commands:
   * Training
     - Training the Attention model:
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
       -vv data/glwordvectors_1_300.txt \
       -eb 300 \
       -cfg config/attention_0.py
       ```
     - Training the Siamese model:
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
   * Evaluating (replace ``XXX`` with the path to the folder that contains the model files)
     - Evaluating the Attention model:
       ```
       python run.py evaluate \
       -de1 data/dat_test_a.conll \
       -de2 data/dat_test_b.conll \
       -ddl data/labels_test.conll \
       -v data/glvocab_1.txt \
       -vv data/glwordvectors_1_300.txt \
       -eb 300 \
       -m attention \
       -cfg config/attention_0.py \
       -mp results/lstm/XXX/
       ```
     - Evaluating the Siamese model:
       ```
       python run.py evaluate \
       -de1 data/dat_test_a.conll \
       -de2 data/dat_test_b.conll \
       -ddl data/labels_test.conll \
       -v data/glvocab_1.txt \
       -vv data/glwordvectors_1_300.txt \
       -eb 300 \
       -m siamese \
       -cfg config/siamese_0.py \
       -mp results/lstm/XXX/
       ```
   * Running the shell (replace ``XXX`` with the path to the folder that contains the model files)
     - Running the shell for the Attention model:
       ```
       python run.py shell \
       -v data/glvocab_1.txt \
       -vv data/glwordvectors_1_300.txt \
       -eb 300 \
       -m attention \
       -cfg config/attention_0.py \
       -mp results/lstm/XXX/
       ```
     - Running the shell for the Siamese model:
       ```
       python run.py shell \
       -v data/glvocab_1.txt \
       -vv data/glwordvectors_1_300.txt \
       -eb 300 \
       -m siamese \
       -cfg config/siamese_0.py \
       -mp results/lstm/XXX/
       ```
   * Running sendEmail.py script:
     ```
     python scripts/sendEmail.py elkhan.dadashov@gmail.com /home/elkhand/QuoraDuplicates/exp_march9_relu_pred.txt "exp1 beta=xx"
     
     python scripts/sendEmail.py <your_email> <path_to_file_to_be_attached_to_email> <some_string_to_be_attached_to_subject>
     ```

   * Getting best result from log file:
     ```
     python scripts/getBestF1ScoreFromLogs.py results/lstm/XXX/log
     ```

## Proposal Google Doc:

https://docs.google.com/document/d/1b9ItoYLbtbE02FOlK1V5eEH_6HvlvUDF_X-QHQPzDoo/edit?usp=sharing 

## Quora Dataset Details:

https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs

## Related Work

- Reasoning about Entailment with Neural Attention https://arxiv.org/pdf/1509.06664v1.pdf
- Siamese Recurrent Architectures for Learning Sentence Similarity http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf 
- Efficient mining of hard examples for object detection http://www.idiap.ch/~fleuret/SMLD/2014/SMLD2014_-_Olivier_Canevet_-_Efficient_mining_of_hard_examples_for_object_detection.pdf 
