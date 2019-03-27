"""Utilities for trainer binary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import types
from collections import Iterable  

# Dependency imports
import time
import numpy as np
import six
from six.moves import input
from six.moves import xrange
from six.moves import zip

import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import saver
from tensorflow.python.training import basic_session_run_hooks

from utils import text_reader
from utils import parallel
from models import common_hparams
from models.transformer import Transformer
from utils import inference


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "", "Base output directory for run.")
flags.DEFINE_string("data_dir", "/tmp/data", "Directory with training data.")
flags.DEFINE_string("train_src_name", "2m.bpe.unk.zh", "src name of training data.")
flags.DEFINE_string("train_tgt_name", "2m.bpe.unk.en", "tgt name of training data.")
flags.DEFINE_string("vocab_src_name", "vocab.bpe.zh", "src name of vocab.")
flags.DEFINE_string("vocab_tgt_name", "vocab.bpe.en", "tgt name of vocab.")
flags.DEFINE_integer("vocab_src_size", 30720, "source vocab size.")
flags.DEFINE_integer("vocab_tgt_size", 30720, "target vocab size.")

flags.DEFINE_string("model", "Transformer", "Which model to use.")
flags.DEFINE_string("hparams_set", "", "Which parameters to use.")
flags.DEFINE_string("hparams_range", "", "Parameters range.")
flags.DEFINE_string("hparams", "", """A comma-separated list of `name=value` hyperparameter values.""")
flags.DEFINE_integer("train_steps", 250000, "The number of steps to run training for.")
flags.DEFINE_integer("eval_steps", 10, "Number of steps in evaluation.")
flags.DEFINE_bool("eval_print", False, "Print eval logits and predictions.")
flags.DEFINE_integer("keep_checkpoint_max", 20, "How many recent checkpoints to keep.")
flags.DEFINE_integer("save_checkpoint_secs", 0, "How seconds to save checkpoints.")
flags.DEFINE_integer("save_checkpoint_steps", 1000, "How steps tp save checkpoints.")
flags.DEFINE_float("gpu_mem_fraction", 0.95, "How GPU memory to use.")
flags.DEFINE_bool("experimental_optimize_placement", False,
                "Optimize ops placement with experimental session options.")

# Distributed training flags
flags.DEFINE_string("master", "", "Address of TensorFlow master.")
flags.DEFINE_string("schedule", "local_run",
                "Method of tf.contrib.learn.Experiment to run.")
flags.DEFINE_bool("locally_shard_to_cpu", False,
                "Use CPU as a sharding device runnning locally. This allows "
                "to test sharded model construction on a machine with 1 GPU.")
flags.DEFINE_bool("daisy_chain_variables", True,
                "copy variables around in a daisy chain")
flags.DEFINE_integer("worker_gpu", 1, "How many GPUs to use.")
flags.DEFINE_integer("worker_replicas", 1, "How many workers to use.")
flags.DEFINE_integer("worker_id", 0, "Which worker task are we.")
flags.DEFINE_string("gpu_order", "", "Optional order for daisy-chaining gpus."
                " e.g. \"1 3 2 4\"")

# Decode flags
flags.DEFINE_string("decode_from_file", None, "Path to decode file")
flags.DEFINE_string("decode_to_file", None, "Path to inference output file")
flags.DEFINE_integer("decode_extra_length", 50, "Added decode length.")
flags.DEFINE_integer("decode_batch_size", 32, "Batch size for decoding. ")
flags.DEFINE_integer("decode_beam_size", 4, "The beam size for beam decoding")
flags.DEFINE_float("decode_alpha", 0.6, "Alpha for length penalty")
flags.DEFINE_bool("decode_return_beams", False,"whether return all beams or one")


def create_hparams():
    """Returns hyperparameters, including any flag value overrides.
    """
    if FLAGS.hparams_set == "transformer_params_base":
        hparams = common_hparams.transformer_params_base(FLAGS.data_dir, FLAGS.vocab_src_name, FLAGS.vocab_tgt_name) ## !!
    elif FLAGS.hparams_set == "transformer_params_big":
        hparams = common_hparams.transformer_params_big(FLAGS.data_dir, FLAGS.vocab_src_name, FLAGS.vocab_tgt_name) ## !!
    else:
        raise ValueError("Do not have right model params")
    hparams.vocab_src_size = FLAGS.vocab_src_size
    hparams.vocab_tgt_size = FLAGS.vocab_tgt_size

    if FLAGS.hparams:
        hparams = hparams.parse(FLAGS.hparams)

    return hparams

def run(data_dir, model, output_dir, train_steps):
    """Runs an Estimator locally or distributed.
    """

    #Build Params
    tf.logging.info("Build Params...")
    hparams = create_hparams()

    if FLAGS.train_steps == 0:
        tf.logging.info("Prepare for Inference...")
        inference_run(model, hparams, output_dir)
        return

    tf.logging.info("Prepare for Training...")
    num_datashards = data_parallelism().n

    # Build Data
    tf.logging.info("Build Data...")
    train_input_fn = get_input_fn(
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        hparams=hparams,
        data_file_patterns=get_datasets_for_mode(data_dir, tf.contrib.learn.ModeKeys.TRAIN),
        num_datashards=num_datashards)
    
    # Build Model
    tf.logging.info("Build Model...")
    model_fn=model_builder(model, hparams=hparams)
    
    # Build Graph
    tf.logging.info("Build Graph...")
    all_hooks = []
    with ops.Graph().as_default() as g:
        global_step = tf.train.create_global_step(g)
        features, labels = train_input_fn()
        model_fn_ops = model_fn(features, labels) #total_loss, train_op 
        ops.add_to_collection(ops.GraphKeys.LOSSES, model_fn_ops[0])

        saver = tf.train.Saver(
        sharded=True,
        max_to_keep=FLAGS.keep_checkpoint_max,
        defer_build=True,
        save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
  
        all_hooks.extend([
            tf.train.StopAtStepHook(last_step=FLAGS.train_steps),
            tf.train.NanTensorHook(model_fn_ops[0]),
            tf.train.LoggingTensorHook(
            {
                'loss': model_fn_ops[0],
                'step': global_step
            },
            every_n_iter=100),
            tf.train.CheckpointSaverHook(
                checkpoint_dir=output_dir,
                save_secs=FLAGS.save_checkpoint_secs or None,
                save_steps=FLAGS.save_checkpoint_steps or None,
                saver=saver) 
        ])
    
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=output_dir,
                hooks=all_hooks,
                save_checkpoint_secs=0,  # Saving is handled by a hook.
                config=session_config(gpu_mem_fraction=FLAGS.gpu_mem_fraction)) as mon_sess:
            loss = None
            while not mon_sess.should_stop():
                _, loss = mon_sess.run([model_fn_ops[1], model_fn_ops[0]])
        return loss


def inference_run(model, hparams, output_dir):

    # Build Model
    tf.logging.info("Build Model...")
    model_fn_inference=model_builder_inference(model, hparams=hparams)

    # Build Graph
    tf.logging.info("Build Graph...")
    checkpoint_path = saver.latest_checkpoint(output_dir)
    if not checkpoint_path:
        raise NotFittedError("Couldn't find trained model at %s."
                % output_dir)
    
    with ops.Graph().as_default() as g:
        tf.train.create_global_step(g)
        inputs_ph = tf.placeholder(tf.int32,[None,None]) ## batch_size
        features = {"inputs":inputs_ph}
        labels = None
        infer_ops = model_fn_inference(features, labels) # predictions, None, None
        predictions = infer_ops[0]
        mon_sess = tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(
                checkpoint_filename_with_path=checkpoint_path,
                config=session_config(gpu_mem_fraction=FLAGS.gpu_mem_fraction)))

    def predict_func(feed_fn=None):
        with ops.Graph().as_default() as g:
            inputs = feed_fn["inputs"]
            feed = {inputs_ph: inputs}
            preds = mon_sess.run(predictions, feed) 

            first_tensor = list(preds.values())[0]
            batch_length = first_tensor.shape[0]
            for i in range(batch_length):
                yield {key: value[i] for key, value in six.iteritems(preds)}

    tf.logging.info("Begin Decoding...")
    inference.decode_from_file(predict_func, hparams, FLAGS.decode_from_file, 
            FLAGS.decode_to_file, 
            FLAGS.decode_batch_size, 
            FLAGS.decode_beam_size,
            FLAGS.decode_return_beams)

def model_builder_inference(model, hparams):

    def model_fn(features, targets):

        mode = tf.contrib.learn.ModeKeys.INFER
        features = _decode_input_tensor_to_features_dict(features, hparams)
        dp = data_parallelism()

        model_class = Transformer(hparams, mode, dp)

        result_list = model_class.infer(
            features,
            beam_size=FLAGS.decode_beam_size,
            top_beams=(FLAGS.decode_beam_size if FLAGS.decode_return_beams else 1),
            alpha=FLAGS.decode_alpha,
            decode_length=FLAGS.decode_extra_length)
  
        if not isinstance(result_list, dict): ## greedy
            ret = {"outputs": result_list}, None, None
        else: ## beam
            ret = {
                "outputs": result_list["outputs"],
                "scores": result_list["scores"]
            }, None, None
        if "inputs" in features:
            ret[0]["inputs"] = features["inputs"]
        if "infer_targets" in features:
            ret[0]["targets"] = features["infer_targets"]

        return ret

    return model_fn
  
def validate_flags():
    if not FLAGS.model:
        raise ValueError("Must specify a model with --model.")
    if not (FLAGS.hparams_set or FLAGS.hparams_range):
        raise ValueError("Must specify either --hparams_set or --hparams_range.")
    if not FLAGS.schedule:
        raise ValueError("Must specify --schedule.")
    if not FLAGS.output_dir:
        FLAGS.output_dir = "/tmp/tensor2tensor"
        tf.logging.warning("It is strongly recommended to specify --output_dir. "
                       "Using default output_dir=%s.", FLAGS.output_dir)


def session_config(gpu_mem_fraction=0.95):
    """The TensorFlow Session config to use."""
    graph_options = tf.GraphOptions(optimizer_options=tf.OptimizerOptions(
        opt_level=tf.OptimizerOptions.L1, do_function_inlining=False))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction)
    config = tf.ConfigProto(
        allow_soft_placement=True, graph_options=graph_options, gpu_options=gpu_options)

    return config


def model_builder(model, hparams):

    def model_fn(features, targets): 
        mode = tf.contrib.learn.ModeKeys.TRAIN
        features["targets_l2r"] = targets
        dp = data_parallelism()
        tf.get_variable_scope().set_initializer(initializer(hparams))
    
        def get_model():
            """Build the model for the n-th problem, plus some added variables."""
            model_class = Transformer(hparams, mode, dp) ##!!!!
            sharded_logits, training_loss, extra_loss = model_class.model_fn(features)

            with tf.variable_scope("losses_avg", reuse=True):
                loss_moving_avg = tf.get_variable("training_loss")
                o1 = loss_moving_avg.assign(loss_moving_avg * 0.9 + training_loss * 0.1)
                loss_moving_avg = tf.get_variable("extra_loss")
                o2 = loss_moving_avg.assign(loss_moving_avg * 0.9 + extra_loss * 0.1)
                loss_moving_avg = tf.get_variable("total_loss")
                total_loss = training_loss + extra_loss
                o3 = loss_moving_avg.assign(loss_moving_avg * 0.9 + total_loss * 0.1)
            with tf.variable_scope("train_stats"):  # Count steps for this problem.
                problem_steps = tf.get_variable(
                    "steps", initializer=0, trainable=False)
                o4 = problem_steps.assign_add(1)
            with tf.control_dependencies([o1, o2, o3, o4]):  # Make sure the ops run.
                # Ensure the loss is a scalar here.
                total_loss = tf.reshape(total_loss, [], name="total_loss_control_id")
            return [total_loss] + sharded_logits    # Need to flatten for cond later.

        result_list = get_model()
        sharded_logits, total_loss = result_list[1:], result_list[0]

        #Some training statistics.
        with tf.name_scope("training_stats"):
            learning_rate = hparams.learning_rate * learning_rate_decay(hparams)
            learning_rate /= math.sqrt(float(FLAGS.worker_replicas))
            tf.summary.scalar("learning_rate", learning_rate)
            global_step = tf.to_float(tf.train.get_global_step())

        # Log trainable weights and add decay.
        total_size, weight_decay_loss = 0, 0.0
        all_weights = {v.name: v for v in tf.trainable_variables()}
        for v_name in sorted(list(all_weights)):
            v = all_weights[v_name]
            v_size = int(np.prod(np.array(v.shape.as_list())))
            tf.logging.info("Weight  %s\tshape    %s\tsize    %d",
                    v.name[:-2].ljust(80), str(v.shape).ljust(20), v_size)
            total_size += v_size
            if hparams.weight_decay > 0.0 and len(v.shape.as_list()) > 1:
            # Add weight regularization if set and the weight is not a bias (dim>1).
                with tf.device(v._ref().device):  # pylint: disable=protected-access
                    v_loss = tf.nn.l2_loss(v) / v_size
                weight_decay_loss += v_loss
            is_body = len(v_name) > 5 and v_name[:5] == "body/"
            if hparams.weight_noise > 0.0 and is_body:
            # Add weight noise if set in hparams.
                with tf.device(v._ref().device):  # pylint: disable=protected-access
                    scale = learning_rate * 0.001
                    noise = tf.truncated_normal(v.shape) * hparams.weight_noise * scale
                    noise_op = v.assign_add(noise)
                with tf.control_dependencies([noise_op]):
                    total_loss = tf.identity(total_loss)
        tf.logging.info("Total trainable variables size: %d", total_size)
        if hparams.weight_decay > 0.0:
            total_loss += weight_decay_loss * hparams.weight_decay
        total_loss = tf.identity(total_loss, name="total_loss")

        # Define the train_op for the TRAIN mode.
        opt = _ConditionalOptimizer(hparams.optimizer, learning_rate, hparams)
        tf.logging.info("Computing gradients for global model_fn.")
        train_op = tf.contrib.layers.optimize_loss(
            name="training",
            loss=total_loss,
            global_step=tf.train.get_global_step(),
            learning_rate=learning_rate,
            clip_gradients=hparams.clip_grad_norm or None,
            optimizer=opt,
            colocate_gradients_with_ops=True)
        tf.logging.info("Global model_fn finished.")
        return total_loss, train_op

    return model_fn

def initializer(hparams):
    if hparams.initializer == "orthogonal":
        return tf.orthogonal_initializer(gain=hparams.initializer_gain)
    elif hparams.initializer == "uniform":
        max_val = 0.1 * hparams.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif hparams.initializer == "normal_unit_scaling":
        return init_ops.variance_scaling_initializer(
            hparams.initializer_gain, mode="fan_avg", distribution="normal")
    elif hparams.initializer == "uniform_unit_scaling":
        return init_ops.variance_scaling_initializer(
            hparams.initializer_gain, mode="fan_avg", distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % hparams.initializer)

def learning_rate_decay(hparams):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(
        hparams.learning_rate_warmup_steps * FLAGS.worker_replicas)
    step = tf.to_float(tf.train.get_global_step())
    if hparams.learning_rate_decay_scheme == "noam":
        return 5000.0 * hparams.hidden_size**-0.5 * tf.minimum(
            (step + 1) * warmup_steps**-1.5, (step + 1)**-0.5)
    else:
        raise ValueError("Unrecognized learning rate decay scheme: %s" %
                hparams.learning_rate_decay_scheme)

def get_datasets_for_mode(data_dir, mode):
    return text_reader.get_datasets(data_dir, mode)


def _decode_input_tensor_to_features_dict(feature_map, hparams):
    """Convert the interactive input format (see above) to a dictionary.
    """
    inputs = tf.convert_to_tensor(feature_map["inputs"])    
    x = tf.expand_dims(inputs, axis=[2])
    x = tf.to_int32(x)

    features = {}
    features["decode_length"] = (tf.shape(x)[1] + 50)
    features["inputs"] = x
    return features


def get_input_fn(mode,
                hparams,
                data_file_patterns=None,
                num_datashards=None,
                fixed_problem=None):
    """Provides input to the graph, either from disk or via a placeholder.
    """
    def input_fn():
        batches = []
        with tf.name_scope("input_queues"):
            #for n in xrange(1):
                #with tf.name_scope("problem_%d" % n):
            with tf.device("/cpu:0"):    # Input queues are on CPU.
                capacity = hparams.max_expected_batch_size_per_shard
                capacity *= num_datashards
                examples = text_reader.input_pipeline(data_file_patterns[0], capacity, mode)
                drop_long_sequences = mode == tf.contrib.learn.ModeKeys.TRAIN
                batch_size_multiplier = hparams.batch_size_multiplier
                feature_map = text_reader.batch_examples(
                    examples,
                    text_reader.hparams_to_batching_scheme(
                        hparams,
                        shard_multiplier=num_datashards,
                        drop_long_sequences=drop_long_sequences,
                        length_multiplier=batch_size_multiplier))

            # Ensure inputs and targets are proper rank.
            while len(feature_map["inputs"].get_shape()) != 4:
                feature_map["inputs"] = tf.expand_dims(feature_map["inputs"], axis=-1)
            while len(feature_map["targets_l2r"].get_shape()) != 4:
                feature_map["targets_l2r"] = tf.expand_dims(feature_map["targets_l2r"], axis=-1)
            while len(feature_map["targets_r2l"].get_shape()) != 4:
                feature_map["targets_r2l"] = tf.expand_dims(feature_map["targets_r2l"], axis=-1)

            batches.append((feature_map["inputs"], feature_map["targets_l2r"], feature_map["targets_r2l"] ))

        # We choose which problem to process.
        loss_moving_avgs = []  # Need loss moving averages for that.
        #for n in xrange(1):
        with tf.variable_scope("losses_avg"):
            loss_moving_avgs.append(
                    tf.get_variable("total_loss", initializer=100.0, trainable=False))
            tf.get_variable("training_loss", initializer=100.0, trainable=False)
            tf.get_variable("extra_loss", initializer=100.0, trainable=False)
    
        rand_inputs, rand_target_l2r, rand_target_r2l = batches[0]

        # Set shapes so the ranks are clear.
        rand_inputs.set_shape([None, None, None, None])
        rand_target_l2r.set_shape([None, None, None, None])
        rand_target_r2l.set_shape([None, None, None, None])

        # Final feature map.
        rand_feature_map = {"inputs": rand_inputs, "targets_r2l": rand_target_r2l}
        return rand_feature_map, rand_target_l2r

    return input_fn


class _ConditionalOptimizer(tf.train.Optimizer):
    """Conditional optimizer."""

    def __init__(self, optimizer_name, lr, hparams):

        if optimizer_name == "Adam":
        # We change the default epsilon for Adam and re-scale lr.
        # Using LazyAdam as it's much faster for large vocabulary embeddings.
            self._opt = tf.contrib.opt.LazyAdamOptimizer(
                lr / 500.0,
                beta1=hparams.optimizer_adam_beta1,
                beta2=hparams.optimizer_adam_beta2,
                epsilon=hparams.optimizer_adam_epsilon)
        elif optimizer_name == "Momentum":
            self._opt = tf.train.MomentumOptimizer(
                lr, momentum=hparams.optimizer_momentum_momentum)
        else:
            self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

    def compute_gradients(self, loss, var_list, colocate_gradients_with_ops):
        return self._opt.compute_gradients(
                loss, var_list, colocate_gradients_with_ops=colocate_gradients_with_ops)

    def apply_gradients(self, gradients, global_step=None, name=None):
        return self._opt.apply_gradients(
                gradients, global_step=global_step, name=name)


def _gpu_order(num_gpus):
    if FLAGS.gpu_order:
        ret = [int(s) for s in FLAGS.gpu_order.split(" ")]
        if len(ret) == num_gpus:
            return ret
    return list(range(num_gpus))


def data_parallelism(all_workers=False):
    """Over which devices do we split each training batch.
    """

    if FLAGS.schedule == "local_run":
        #assert not FLAGS.sync
        datashard_devices = ["gpu:%d" % d for d in _gpu_order(FLAGS.worker_gpu)]
        if FLAGS.locally_shard_to_cpu:
            datashard_devices += ["cpu:0"]
        caching_devices = None
    
    tf.logging.info("datashard_devices: %s", datashard_devices)
    tf.logging.info("caching_devices: %s", caching_devices)
    return parallel.Parallelism(
            datashard_devices,
            reuse=True,
            caching_devices=caching_devices,
            daisy_chain_variables=FLAGS.daisy_chain_variables)


