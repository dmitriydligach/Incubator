#!/usr/bin/env python3

import glob, collections, os

class Tokenizer:
  """Tokenization and vectorization"""

  def __init__(self, n_words, lower=False):
    """Constructiion deconstruction"""

    self.n_words = n_words
    self.lower = lower
    self.stoi = {}
    self.itos = {}

  def fit_on_texts(self, texts):
    """Fit on a list of documents"""

    counts = collections.Counter()

    for text in texts:
      tokens = text.split()
      counts.update(tokens)

    # zero is reserved for something
    index = 1

    for token, _ in counts.most_common(self.n_words):
      self.stoi[token] = index
      self.itos[index] = token
      index += 1

class DirDataReader:
  """Each label is in a subdirectory"""

  @staticmethod
  def read(path, label2int):
    """Subdirectories are class labels"""

    labels = []  # int labels
    samples = [] # examples as strings

    for label_dir in os.listdir(path):
      label_dir_path = os.path.join(path, label_dir)

      for file in os.listdir(label_dir_path):
        file_path = os.path.join(label_dir_path, file)
        file_text = open(file_path).read().rstrip()
        int_label = label2int[label_dir.lower()]
        samples.append(file_text)
        labels.append(int_label)

    return samples, labels

if __name__ == "__main__":

  samples, labels = DirDataReader.read(
    '/Users/Dima/Work/Data/Opioids/Train/',
    {'no':0, 'yes':1})

  tokenizer = Tokenizer(n_words=100)
  tokenizer.fit_on_texts(samples)
  print(tokenizer.stoi)
  print(tokenizer.itos)