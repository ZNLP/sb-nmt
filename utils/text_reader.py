"""Encoders for text data.

* TextEncoder: base class
* ByteTextEncoder: for ascii text
* TokenTextEncoder: with user-supplied vocabulary file
* SubwordTextEncoder: invertible
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

# Dependency imports

import six
import math
from six import PY2
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import os

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import tensorflow as tf

#from models import common_hparams
from models import common_layers


# Conversion between Unicode and UTF-8, if required (on Python2)
native_to_unicode = (lambda s: s.decode("utf-8")) if PY2 else (lambda s: s)
unicode_to_native = (lambda s: s.encode("utf-8")) if PY2 else (lambda s: s)


# Reserved tokens for things like padding and EOS symbols.
PAD = "<pad>"
EOS = "<EOS>"
RESERVED_TOKENS = [PAD, EOS]
if six.PY2:
    RESERVED_TOKENS_BYTES = RESERVED_TOKENS
else:
    RESERVED_TOKENS_BYTES = [bytes(PAD, "ascii"), bytes(EOS, "ascii")]


class TextEncoder(object):
    """Base class for converting from ints to/from human readable strings."""

    def __init__(self, num_reserved_ids=2):
        self._num_reserved_ids = num_reserved_ids

    def encode(self, s):

        """Transform a human-readable string into a sequence of int ids."""
        return [int(w) + self._num_reserved_ids for w in s.split()]

    def decode(self, ids):
        """Transform a sequence of int ids into a human-readable string."""

        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < self._num_reserved_ids:
                decoded_ids.append(RESERVED_TOKENS[int(id_)])
            else:
                decoded_ids.append(id_ - self._num_reserved_ids)
        return " ".join([str(d) for d in decoded_ids])

    @property
    def vocab_size(self):
        raise NotImplementedError()


class TokenTextEncoder(TextEncoder):
    """Encoder based on a user-supplied vocabulary."""

    def __init__(self, vocab_filename, reverse=False, num_reserved_ids=2):
        """Initialize from a file, one token per line."""
        super(TokenTextEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._load_vocab_from_file(vocab_filename)

    def encode(self, sentence, replace_oov=None):
        """Converts a space-separated string of tokens to a list of ids."""
        tokens = sentence.strip().split()
        if replace_oov is not None:
            tokens = [t if t in self._token_to_id else replace_oov for t in tokens]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret

    def decode(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return " ".join([self._safe_id_to_token(i) for i in seq])

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" % idx)

    def _load_vocab_from_file(self, filename):
        """Load vocab from a file."""
        self._token_to_id = {}
        self._id_to_token = {}

        for idx, tok in enumerate(RESERVED_TOKENS):
            self._token_to_id[tok] = idx
            self._id_to_token[idx] = tok

        token_start_idx = self._num_reserved_ids
        with tf.gfile.Open(filename) as f:
            for i, line in enumerate(f):
                idx = token_start_idx + i
                tok = line.strip()
                self._token_to_id[tok] = idx
                self._id_to_token[idx] = tok


###########################################################
def examples_queue(data_sources,
                   data_fields_to_features,
                   training,
                   capacity=32,
                   data_items_to_decoders=None,
                   data_items_to_decode=None):
    """Contruct a queue of training or evaluation examples.
    """
    with tf.name_scope("examples_queue"):
        # Read serialized examples using slim parallel_reader.
        num_epochs = None if training else 1
        data_files = tf.contrib.slim.parallel_reader.get_data_files(data_sources)
        num_readers = min(4 if training else 1, len(data_files))
        _, example_serialized = tf.contrib.slim.parallel_reader.parallel_read(
            data_sources,
            tf.TFRecordReader,
            num_epochs=num_epochs,
            shuffle=training,
            capacity=2 * capacity,
            min_after_dequeue=capacity,
            num_readers=num_readers)

        if data_items_to_decoders is None:
            data_items_to_decoders = {
            field: tf.contrib.slim.tfexample_decoder.Tensor(field)
            for field in data_fields_to_features
        }

        decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
            data_fields_to_features, data_items_to_decoders)

        if data_items_to_decode is None:
            data_items_to_decode = list(data_items_to_decoders)

        decoded = decoder.decode(example_serialized, items=data_items_to_decode)
        return {
            field: tensor
            for (field, tensor) in zip(data_items_to_decode, decoded)
        }


def input_pipeline(data_file_pattern, capacity, mode):
    """Input pipeline, returns a dictionary of tensors from queues."""

    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets_l2r": tf.VarLenFeature(tf.int64),
        #"targets": tf.VarLenFeature(tf.int64),
        "targets_r2l": tf.VarLenFeature(tf.int64)}
    data_items_to_decoders = None

    examples = examples_queue(
        [data_file_pattern],
        data_fields,
        training=(mode == tf.contrib.learn.ModeKeys.TRAIN),
        capacity=capacity,
        data_items_to_decoders=data_items_to_decoders)

    # We do not want int64s as they do are not supported on GPUs.
    return {k: tf.to_int32(v) for (k, v) in six.iteritems(examples)}


def batch_examples(examples, batching_scheme):
    """Given a queue of examples, create batches of examples with similar lengths.
    """
    with tf.name_scope("batch_examples"):
        # The queue to bucket on will be chosen based on maximum length.
        max_length = 0
        for v in examples.values():
        # For images the sequence length is the size of the spatial dimensions.
            sequence_length = (tf.shape(v)[0] if len(v.get_shape()) < 3 else
                    tf.shape(v)[0] * tf.shape(v)[1])
            max_length = tf.maximum(max_length, sequence_length)
        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_length,
            examples,
            batching_scheme["batch_sizes"],
            [b + 1 for b in batching_scheme["boundaries"]],
            capacity=2,  # Number of full batches to store, we don't need many.
            bucket_capacities=[2 * b for b in batching_scheme["batch_sizes"]],
            dynamic_pad=True,
            keep_input=(max_length <= batching_scheme["max_length"]))
        return outputs


def bucket_boundaries(max_length, min_length=8, mantissa_bits=2):
    """A default set of length-bucket boundaries."""
    x = min_length
    boundaries = []
    while x < max_length:
        boundaries.append(x)
        x += 2**max(0, int(math.log(x, 2)) - mantissa_bits)
    return boundaries


def hparams_to_batching_scheme(hparams,
                               drop_long_sequences=False,
                               shard_multiplier=1,
                               length_multiplier=1):
    """A batching scheme based on model hyperparameters.
    """
    max_length = hparams.max_length or hparams.batch_size
    boundaries = bucket_boundaries(
        max_length, mantissa_bits=hparams.batching_mantissa_bits)
    batch_sizes = [
        max(1, hparams.batch_size // length)
        for length in boundaries + [max_length]
    ]
    batch_sizes = [b * shard_multiplier for b in batch_sizes]
    max_length *= length_multiplier
    boundaries = [boundary * length_multiplier for boundary in boundaries]
    return {
        "boundaries": boundaries,
        "batch_sizes": batch_sizes,
        "max_length": (max_length if drop_long_sequences else 10**9)
    }


def get_datasets(data_dir, mode):
    """Return the location of a dataset for a given mode."""
    datasets = []
    #for problem in ["translation",]:
    for problem in ["wmt_ende_bpe32k",]:
        #problem, _, _ = common_hparams.parse_problem_name(problem)
        path = os.path.join(data_dir, problem)
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            datasets.append("%s-train*" % path)
        else:
            datasets.append("%s-dev*" % path)
    return datasets

