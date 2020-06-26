#!/usr/bin/env python3

import torch
import numpy as np
from transformers import BertTokenizer
from tokenizers import CharBPETokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

def to_transformer_inputs(seqs, max_len=None):
  """Matrix of token ids and a square attention mask for eash sample"""

  if max_len is None:
    # set max_len to the length of the longest sequence
    max_len = max(len(id_seq) for id_seq in seqs)

  ids = torch.zeros(len(seqs), max_len, dtype=torch.long)
  mask = torch.zeros(len(seqs), max_len, max_len, dtype=torch.long)

  for i, seq in enumerate(seqs):
    if len(seq) > max_len:
      seq = seq[:max_len]
    ids[i, :len(seq)] = torch.tensor(seq)
    mask[i, :len(seq), :len(seq)] = 1

  return ids, mask

def make_data_loader(texts, labels, batch_size, max_len, partition, input_processor):
  """DataLoader objects for train or dev/test sets"""

  model_inputs = input_processor(texts, max_len)
  labels = torch.tensor(labels)

  # e.g. transformers take input ids and attn masks
  if type(model_inputs) is tuple:
    tensor_dataset = TensorDataset(*model_inputs, labels)
  else:
    tensor_dataset = TensorDataset(model_inputs, labels)

  # use sequential sampler for dev and test
  if partition == 'train':
    sampler = RandomSampler(tensor_dataset)
  else:
    sampler = SequentialSampler(tensor_dataset)

  data_loader = DataLoader(
    tensor_dataset,
    sampler=sampler,
    batch_size=batch_size)

  return data_loader

if __name__ == "__main__":

  texts = ['it is happening again',
           'the owls are not what they seem']
  ids, masks = to_transformer_inputs(texts, max_len=None)
  print('ids:', ids)
  print('masks:', masks)