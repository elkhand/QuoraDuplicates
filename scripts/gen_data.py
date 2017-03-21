# Generates glvocab_1.txt and glwordvectors_1_[100/300].txt from the GloVe data.
#
# Download the GloVe data from http://nlp.stanford.edu/data/glove.6B.zip

vocab = None

for embed_size in [100, 300]:
  current_vocab = []

  with open("glove.6B.%dd.txt" % embed_size, "r") as f_in:
    with open("glwordvectors_1_%d.txt" % embed_size, "w") as f_out:
      for line in f_in:
        cols = line.split()
        assert len(cols) == embed_size + 1

        word = cols[0]
        current_vocab.append(word)
        f_out.write(line[len(word)+1:])

  if vocab is None:
    vocab = current_vocab
  else:
    assert vocab == current_vocab

assert len(vocab) == 400000

with open("glvocab_1.txt", "w") as f:
  for word in vocab:
    f.write(word + "\n")
