#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import torch
import torch.nn as nn

from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import os, configparser, math, random
import datareader, tokenizer, utils

# deterministic determinism
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(2020)

class TransformerClassifier(nn.Module):
  """A transformative experience"""

  def __init__(self, num_classes=2 ):
    """We have some of the best constructors in the world"""

    super(TransformerClassifier, self).__init__()

    self.embedding = nn.Embedding(
      num_embeddings=cfg.getint('data', 'vocab_size'),
      embedding_dim=cfg.getint('model', 'emb_dim'))

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=cfg.getint('model', 'emb_dim'),
      nhead=cfg.getint('model', 'num_heads'),
      dim_feedforward=cfg.getint('model', 'feedforw_dim'))

    self.trans_encoder = nn.TransformerEncoder(
      encoder_layer=encoder_layer,
      num_layers=cfg.getint('model', 'num_layers'))

    self.dropout = nn.Dropout(cfg.getfloat('model', 'dropout'))

    self.linear = nn.Linear(
      in_features=cfg.getint('model', 'emb_dim'),
      out_features=num_classes)

  def forward(self, texts, attention_mask):
    """Moving forward"""

    output = self.embedding(texts) # * sqrtn

    # encoder input: (seq_len, batch_size, emb_dim)
    # encoder output: (seq_len, batch_size, emb_dim)
    output = output.permute(1, 0, 2)
    output = self.trans_encoder(output, attention_mask)

    # extract CLS token only
    # output = output[0, :, :]

    # average pooling
    output = torch.mean(output, dim=0)

    output = self.dropout(output)
    output = self.linear(output)

    return output

def train(model, train_loader, val_loader, weights):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  weights = weights.to(device)
  cross_entropy_loss = torch.nn.CrossEntropyLoss(weights)

  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.getfloat('model', 'lr'))

  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000)

  for epoch in range(1, cfg.getint('model', 'num_epochs') + 1):
    model.train()
    train_loss, num_train_steps = 0, 0

    for batch in train_loader:
      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_mask, batch_labels = batch
      batch_mask = batch_mask.repeat(cfg.getint('model', 'num_heads'), 1, 1)
      optimizer.zero_grad()

      logits = model(batch_inputs, batch_mask)
      loss = cross_entropy_loss(logits, batch_labels)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_loss = train_loss / num_train_steps
    val_loss, f1 = evaluate(model, val_loader, weights)
    print('ep: %d, steps: %d, tr loss: %.3f, val loss: %.3f, val roc: %.3f' % \
          (epoch, num_train_steps, av_loss, val_loss, f1))

def evaluate(model, data_loader, weights):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  weights = weights.to(device)
  model.to(device)

  cross_entropy_loss = torch.nn.CrossEntropyLoss(weights)
  total_loss, num_steps = 0, 0

  model.eval()

  all_labels = []
  all_probs = []

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_mask, batch_labels = batch
    batch_mask = batch_mask.repeat(cfg.getint('model', 'num_heads'), 1, 1)

    with torch.no_grad():
      logits = model(batch_inputs, batch_mask)
      loss = cross_entropy_loss(logits, batch_labels)

    batch_logits = logits.detach().to('cpu')
    batch_labels = batch_labels.to('cpu')
    batch_probs = torch.nn.functional.softmax(batch_logits, dim=1)[:, 1]

    all_labels.extend(batch_labels.tolist())
    all_probs.extend(batch_probs.tolist())

    total_loss += loss.item()
    num_steps += 1

  roc_auc = roc_auc_score(all_labels, all_probs)
  return total_loss / num_steps, roc_auc
 
def main():
  """Fine-tune bert"""

  #
  # split train into train and validation and evaluate
  #

  tr_texts, tr_labels = datareader.DirDataReader.read(
    os.path.join(base, cfg.get('data', 'train')),
    {'no':0, 'yes':1})

  tr_texts, val_texts, tr_labels, val_labels = train_test_split(
    tr_texts, tr_labels, test_size=0.15, random_state=2020)

  tok = tokenizer.Tokenizer(cfg.getint('data', 'vocab_size'))
  tok.fit_on_texts(tr_texts)

  tr_texts = tok.texts_as_sets_to_seqs(tr_texts)
  val_texts = tok.texts_as_sets_to_seqs(val_texts)

  train_loader = utils.make_data_loader(
    tr_texts,
    tr_labels,
    cfg.getint('model', 'batch_size'),
    cfg.getint('data', 'max_len'),
    'train',
    utils.to_transformer_inputs)

  val_loader = utils.make_data_loader(
    val_texts,
    val_labels,
    cfg.getint('model', 'batch_size'),
    cfg.getint('data', 'max_len'),
    'dev',
    utils.to_transformer_inputs)

  print('loaded %d training and %d validation samples' % \
        (len(tr_texts), len(val_texts)))

  model = TransformerClassifier()

  label_counts = torch.bincount(torch.IntTensor(tr_labels))
  weights = len(tr_labels) / (2.0 * label_counts)

  train(model, train_loader, val_loader, weights)

  #
  # now retrain and evaluate on test
  #

  tr_texts, tr_labels = datareader.DirDataReader.read(
    os.path.join(base, cfg.get('data', 'train')),
    {'no':0, 'yes':1})

  test_texts, test_labels = datareader.DirDataReader.read(
    os.path.join(base, cfg.get('data', 'test')),
    {'no':0, 'yes':1})

  tok = tokenizer.Tokenizer(cfg.getint('data', 'vocab_size'))
  tok.fit_on_texts(tr_texts)

  tr_texts = tok.texts_as_sets_to_seqs(tr_texts)
  test_texts = tok.texts_as_sets_to_seqs(test_texts)

  train_loader = utils.make_data_loader(
    tr_texts,
    tr_labels,
    cfg.getint('model', 'batch_size'),
    cfg.getint('data', 'max_len'),
    'train',
    utils.to_transformer_inputs)

  test_loader = utils.make_data_loader(
    test_texts,
    test_labels,
    cfg.getint('model', 'batch_size'),
    cfg.getint('data', 'max_len'),
    'test',
    utils.to_transformer_inputs)

  print('loaded %d training and %d test samples' % \
        (len(tr_texts), len(test_texts)))

  model = TransformerClassifier()

  label_counts = torch.bincount(torch.IntTensor(tr_labels))
  weights = len(tr_labels) / (2.0 * label_counts)

  train(model, train_loader, test_loader, weights)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
