"""
Loading and using the Mini-ImageNet dataset.

To use these APIs, you should prepare a directory that
contains three sub-directories: train, test, and val.
Each of these three directories should contain one
sub-directory per WordNet ID.
"""

import os
import random
import h5py
from collections import defaultdict

from PIL import Image
import numpy as np
import random

def read_dataset(data_dir):
    """
    Read the Mini-ImageNet dataset.

    Args:
      data_dir: directory containing Mini-ImageNet.

    Returns:
      A tuple (train, val, test) of sequences of
        ImageNetClass instances.
    """

    embed_file = 'elmo_embeddings.hdf5'
    ex_file = 'examples.tsv'
    dim = 1024

    ex_info = []
    with open(ex_file) as f:
        for line in f:
            e1, conf, _, e1s, e1e, e2, conf2, _, e2s, e2e, example_id, relation, sentence = line.strip().split('\t')
            e1s = int(e1s)
            e1e = int(e1e)
            e2s = int(e2s)
            e2e = int(e2e)
            offsets = (e1e, e2s) if e1s < e2s else (e2e, e1s)
            ex_info.append((relation, sentence, offsets))

    rel_examples = defaultdict(list)
    with h5py.File(os.path.join(data_dir, embed_file), 'r') as f:
        for i, ex in enumerate(ex_info):
            rel, sent, offsets = ex
            # last layer
            last_layer = f[str(i)][-1, :, :]
            s_idx, e_idx = offsets[i]
            if e_idx - s_idx > 1:
                context_vecs = last_layer[s_idx: e_idx]
                avg_vec = np.mean(context_vecs, 0)
            else:
                avg_vec = np.zeros([1, dim])
            avg_vec = np.reshape(avg_vec, [1, -1])
            rel_examples[rel].append((rel, sent, offsets, avg_vec))

    mimls = [MimlClass(rel, ex) for rel, ex in rel_examples.values()]
    return mimls


def split_dataset(all_examples):
    random.shuffle(all_examples)
    size = len(all_examples) / 3
    train = all_examples[:size]
    val = all_examples[size:2 * size]
    test = all_examples[2 * size:]

    return train, val, test


# pylint: disable=R0903
class MimlClass:
    """
    A single image class.
    """
    def __init__(self, rel, examples):
        self.rel = rel
        self.examples = examples

    def sample(self, num_examples):
        """
        Sample images (as numpy arrays) from the class.

        Returns:
          A sequence of 84x84x3 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        random.shuffle(self.examples)
        sampled = []
        for ex in self.examples[:num_examples]:
            sampled.append(ex[-1])
        return sampled

