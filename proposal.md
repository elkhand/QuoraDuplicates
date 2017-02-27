# CS224N Project Proposal

## Team Members
Katherine Yu (yukather), Sukolsak Sakshuwong (sukolsak) , Elkhan Dadashov (elkhand)

## Mentor
Kai Sheng Tai

## Problem Description
Determine if pairs of Quora questions containing similar words are exact duplicates in meaning.  The solution to this problem is interesting to aggregate answers on duplicate questions, and suggest similar questions to users posting new questions on Quora.

## Data
* The data is a public dataset from Quora: [https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs]
* The data contains 400k pairs of questions, 37% are positive examples. The pairs are hand-labeled.
* Example 1: 
  * Question1 - "How can I increase the speed of my internet connection while using a VPN?" 
  * Question2 - "How can Internet speed be increased by hacking through DNS?"
  * This pair is negative, since the idea of hacking through DNS is to increase internet speed is different from increasing speed while you are using VPN.
* Example 2:
  * Question1 - "Why are so many Quora users posting questions that are readily answered on Google?"
  * Question2 - "Why do people ask Quora questions which can be answered easily by Google?"
  * This pair is positive, since the meanings are exactly equivalent up to human judgement: Quora users posting questions is the same as people asking Quora questions, and "readily" is equivalent to "easily."
* We will do a standard train/dev/test split, choosing models based on the dev set, and reporting final evaluation on the test set.

## Methodology/Algorithm
* Baseline1:
  * Tokenize and sum GloVe vectors for questions 1 and 2, take straight cosine similarity between two resulting vectors as score - optimize a threshold for F1 on train set, also look at AUC.
* Baseline2: 
  <!-- 1-layer methods like logistic regression or SVM might not work well without feature engineering relatedness-features (e.g. cosine similarity on tf-idf), which we do not want to spend time on in this class. --> 
  * A Siamese net, with shared parameters W and b,  h1=f(Wx1+b), h2=f(Wx2+b) where x1 and x2 are the summed GloVe vectors for questions 1 and 2, an activation layer between h1 and h2, and cross entropy loss on the output of the activation layer.
* Primary goals:
   * Baseline: Siamese net with summed GloVe vectors
   * LSTM with attention
* Secondary goals:
   * smart undersampling in the majority class by choosing hard examples in mini-batch allocation [http://www.idiap.ch/~fleuret/SMLD/2014/SMLD2014_-_Olivier_Canevet_-_Efficient_mining_of_hard_examples_for_object_detection.pdf]
   * character learning/two-byte encodings

## Related Work

* Basic paper about LSTM with attention (to determine logical entailment): Rocktäschel et al. 100D LSTMs w/ word-by-word attention. [https://arxiv.org/pdf/1509.06664v1.pdf]
   * All of the papers we've seen so far (specfically snli papers) are related to logical inference/entailment classification. We would like to find a paper about determining meaning similarity specifically as a final target.
* Mueller et al. Siamese Recurrent Architectures for Learning Sentence Similarity. [http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf]
   * This paper presents LSTM's for similarity but it seems that their corpus of pairs of sentences are much further away in meaning between pairs than in our dataset.


## Evaluation Plan
* Our final evaluation metric will be F1 on the test set. We will also look at AUC.
* We will produce tuning plots for hyperparameters: 
    * alpha in cross entropy loss to weight positive examples versus negative examples
* We will report test F1 on different methods. We expect these to be competitive.
* We will also look closely at the confusion matrix and do error analysis by hand on sampled example predictions to find insights on where our model is doing well/poorly.

