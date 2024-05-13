
import math
from typing import Dict, List, Optional
import numpy as np

from absl import logging
import tensorflow as tf
import jax
from jax import numpy as jnp
import multihost_dataloading

class PileDatasets():
    def __init__(self,
                mesh: str = None,
                name: str = 'pile',
                path: Optional[str] = None,
                num_infeed_hosts: int = 0,
                reset_for_eval: bool = False,
                batch_size: int = 8,
                seq_len: int = 2048,
                repeat: int = 1,
                seed: int = 9876,
                task_features: Optional[dict] = None,
                shuffle_buffer_size: Optional[int] = None,
                pad_id: int = 0,
                drop_remainder: bool = True,
                iter_file_nums: int = 2, # 100  500 steps/file,
                meta_dict: Optional[dict] = None,
                num_batches_to_skip: Optional[int] = None,
                only_eval: bool = False,
                zero_loss: bool = True,
                ):
        self.mesh = mesh
        self.name = name
        self.path = path
        self.num_infeed_hosts = num_infeed_hosts
        self.reset_for_eval = reset_for_eval
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.repeat = repeat
        self.seed = seed
        self.task_features = task_features
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pad_id = pad_id
        self.drop_remainder = drop_remainder
        self.iter_file_nums = iter_file_nums
        self.meta_dict = meta_dict
        self.num_batches_to_skip = num_batches_to_skip
        self.only_eval = only_eval
        self.zero_loss = zero_loss
        self.batch_padding_size = 0
        self.__post_init__()
        
    def __post_init__(self):
        if self.num_infeed_hosts == 0:
            self.num_infeed_hosts = jax.process_count()

        if not self.meta_dict or self.only_eval:
            self.meta_dict = {}
            self.init_meta()
        else:
            if self.meta_dict["file_in_data"] != 0:
                assert self.meta_dict["iter_file_nums"] == self.iter_file_nums, print(
                    f'iter_file_nums in meta_dict is not equal to cur args. => {self.meta_dict["iter_file_nums"]}≠'
                    f" {self.iter_file_nums}"
                )
            self.step_in_file = self.meta_dict.get('step_in_file')  # XD fix

        if self.num_batches_to_skip is not None:
            self.step_in_file = self.num_batches_to_skip

        logging.info(f'meta_dict: {self.meta_dict}')
        self.seed = self.meta_dict['seed']
        self.dataset = self.load_tfrecord_dataset(fnames=self.path)
        self._peek = None
        self._state_before_peek = None


    def init_meta(self):
        self.meta_dict = {
                "seed": self.seed,
                "cur_files": self.meta_dict.get('cur_files', []),
                "file_in_data": 0,
                "step_in_file": 0,
                "iter_file_nums": self.iter_file_nums,
                "checkpoint_step": self.meta_dict.get('checkpoint_step', None),
            }
        self.step_in_file = 0

 #   def peek_padded(self):
  #      return self.get_next_padded()

    def reset(self):
        self.init_meta()
        self.dataset = self.load_tfrecord_dataset(fnames=self.path)

    def __iter__(self):
        return self.get_next_padded()
    
    def __next__(self):
        return self.get_next_padded()

    def get_next_padded(self):
        if self._peek is not None:
          output = self._peek
          self._peek = None
          self._state_before_peek = None
          return output
        unpadded = next(self.dataset)
        pad_size = int(self.batch_padding_size)
        if pad_size == 0:
            return unpadded
        return jax.tree_util.tree_map(
            lambda x: np.pad(x, [[0, pad_size]] + [[0, 0]] * (x.ndim - 1)),
            unpadded,
        )

    def get_global_batch_size(self, train_input):
        # logging.info(f"train_input: {train_input} type: {type(train_input)}")
        return self.batch_size * self.num_infeed_hosts

    def _parse_function(self, example_proto):
        feature_desc = {key: tf.io.VarLenFeature(tf.int64) for key in self.task_features}
        example = tf.io.parse_single_example(example_proto, feature_desc)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = tf.sparse.to_dense(t, default_value=0)[ :self.seq_len]
        return example


#                        data['inputs'],
#                        data['inputs_position'],
#                        decoder_segment_ids=data['inputs_segmentation'],
                       
#                        enable_dropout=config.enable_dropout if is_train else False,
#                        rngs={'dropout': rng1, 'params': aqt_rng}, mutable='intermediates')
#   one_hot_targets = jax.nn.one_hot(data['targets'], config.vocab_size)
#   xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
#   xent = nn.with_logical_constraint(xent, ('activation_batch', 'activation_length'))
#   # Mask out paddings at the end of each example.
#   xent = xent * (data['targets_segmentation'] != 0)

    def convert(self, data):
        seq_len = self.seq_len
        model_needed_inputs = {}
        model_needed_inputs['inputs'] = data["input_ids"][:, : seq_len - 1]
        model_needed_inputs['targets'] = data["input_ids"][:, 1: seq_len]
        key = 'labels' if "labels" in data else 'input_ids'
        weights = data[key] >= 0 if self.zero_loss else data[key] > 0
        model_needed_inputs['targets_segmentation'] = weights[:, :seq_len - 1]
        model_needed_inputs['inputs_segmentation'] = tf.ones_like(model_needed_inputs['inputs'])
        pos = tf.range(seq_len - 1)
        model_needed_inputs['inputs_position'] = model_needed_inputs['inputs_segmentation'] * pos
        return model_needed_inputs

    def _load_file_dataset(self, fname):
        tf.random.set_seed(self.seed)
        ds = tf.data.Dataset.from_tensor_slices(fname)
        ds = ds.apply(tf.data.TFRecordDataset)
        # shard host data
        process_index = jax.process_index()
        # 在这里进行shard的话，不同的pod在相同的batch_size时，拿到的数据不一致
        ds = ds.shard(self.num_infeed_hosts, process_index)
        # logging.info(f"num_infeed_hosts: {self.num_infeed_hosts} || process_index: {process_index}")  # XD fix
        ds = ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        if self.shuffle_buffer_size is not None:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)
        padded_shapes = {key: self.seq_len for key in self.task_features}
        padding_values = {key: self.pad_id for key in self.task_features}
        ds = ds.padded_batch(
            batch_size=np.prod(self.batch_size),
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=True,
        )
        # lsp: batch之后进行shard。如果不进行shuffle，在batch化之前shard也行
        # ds = ds.shard(self.num_infeed_hosts, process_index)
        ds = ds.map(self.convert)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        if self.step_in_file: ds = ds.skip(self.step_in_file)  # XD fix

        ds = multihost_dataloading.MultiHostDataLoadIterator(ds, self.mesh)

        return ds

    def load_tfrecord_dataset(self, fnames):
        tf.random.set_seed(self.seed)
        assert isinstance(fnames, list)
        repeat_fnames = fnames * self.repeat
        N = math.ceil(len(repeat_fnames) / self.iter_file_nums)
        file_in_data = self.meta_dict["file_in_data"]
        logging.info(f'file_in_data: {file_in_data} N: {N}')
        for n in range(file_in_data, N, 1):
            fname = repeat_fnames[n * self.iter_file_nums : (n + 1) * self.iter_file_nums]
            self.meta_dict["cur_files"] = fname
            ds = self._load_file_dataset(fname)
            # ds = ds.as_numpy_iterator()
            for batch in ds:
                # self.meta_dict["step_in_file"] += 1  # XD fix
                self.step_in_file += 1
                yield batch
            self.meta_dict["file_in_data"] += 1
            # self.meta_dict["step_in_file"] = 0  # XD fix
            self.step_in_file = 0
