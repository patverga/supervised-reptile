"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)

# pylint: disable=R0903
class OmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)


# pylint: disable=R0903
class MiniImageNetModel:
    """
    A model for Mini-ImageNet classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3))
        out = self.input_ph
        for _ in range(4):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)


class ElmoMimlModel:
    """
    A model for Mini-ImageNet classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        dim = 1024
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 3, dim))

        out = self.avg_layers(self.input_ph)

        for _ in range(2):
            dim /= 2
            out = tf.layers.dense(out, dim, activation=tf.nn.relu)
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)

    def avg_layers(self, layers, n_lm_layers=3):
        layer_weight = tf.get_variable(
            'ELMo_W',
            shape=3,
            initializer=tf.zeros_initializer,
            trainable=True,
        )
        normed_weights = tf.split(
            tf.nn.softmax(layer_weight + 1.0 / n_lm_layers), n_lm_layers
        )
        # split LM layers
        layers = tf.split(layers, n_lm_layers, axis=1)

        # compute the weighted, normalized LM activations
        pieces = []
        for w, t in zip(normed_weights, layers):
            pieces.append(w * tf.squeeze(t, squeeze_dims=1))
        sum_pieces = tf.add_n(pieces)
        return sum_pieces
