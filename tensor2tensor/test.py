# -*- coding: utf-8 -*-
import numpy as np


def test_get_weigts():
  """
    layers/modalities.py: get_weigts()
  """
  vocab_size = 8236
  num_shards = 16
  shard_size = []
  for i in range(16):
    shard_size.append((vocab_size // num_shards) +
                      (1 if i < vocab_size % num_shards else 0))


def test_tf_reduce():
  """

  modalities.py classlabel top():
  x = body_output
  x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
  """
  raw = np.load("./test_sg/raw.npy")
  reduced = np.load("./test_sg/redu.npy")

  i = 10
  j = 1
  k = 120
  entry0 = raw[i, :, 0, 0]
  entry0 = np.mean(entry0)
  print(np.allclose(entry0, reduced[i, 0, 0, 0]))
