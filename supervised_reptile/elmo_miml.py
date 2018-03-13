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
import codecs
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
    min_count = 15

    NULL_RELATION='no_relation'
    filter_null = False

    ex_info = []
    with codecs.open(os.path.join(data_dir, ex_file), encoding='utf-8', errors='ignore') as f:
        for line in f:
            e1, conf, _, e1s, e1e, e2, conf2, _, e2s, e2e, example_id, relation, sentence = line.strip().split('\t')
            if not filter_null or relation != NULL_RELATION:
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
            # last_layer = f[str(i)][-1, :, :]
            last_layer = f[str(i)]
            s_idx, e_idx = offsets
            if e_idx - s_idx > 1:
                context_vecs = last_layer[:, s_idx:e_idx, :]
                avg_vec = np.mean(context_vecs, 1)
            else:
                avg_vec = np.zeros([3, dim])
            avg_vec = np.reshape(avg_vec, [3, -1])
            rel_examples[rel].append((rel, sent, offsets, avg_vec))

    mimls = [MimlClass(rel, ex) for rel, ex in rel_examples.items()]
    mimls = [m for m in mimls if len(m.examples) > min_count]
    return mimls


def split_dataset(all_examples):
    random.shuffle(all_examples)
    size = int(len(all_examples) / 3)
    train = all_examples[:size*2]
    test = all_examples[2 * size:]

    print('Train classes: %d' % len(train))
    print('Test classes: %d' % len(test))

    return train, test


# pylint: disable=R0903
class MimlClass:
    """
    A single image class.
    """
    def __init__(self, rel, examples):
        self.rel = rel
        self.examples = examples
        print('%s: %d' % (rel, len(examples)))

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

