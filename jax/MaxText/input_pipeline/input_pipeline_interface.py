"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""Input pipeline"""

import os

import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from input_pipeline import _tfds_data_processing
from input_pipeline import _grain_data_processing
from input_pipeline import _tfds_data_processing_c4_mlperf
from input_pipeline import _pile_data_processing

import tokenizer

import json

import tensorflow as tf
import numpy as np
from google.cloud import storage

from absl import logging



def get_tokenizer(tokenizer_path, add_bos=True, add_eos=True):
  # Load tokenizer
  sp_tokenizer = tokenizer.load_tokenizer(tokenizer_path=tokenizer_path,
                                          add_bos=add_bos,
                                          add_eos=add_eos)
  return sp_tokenizer

def make_c4_mlperf_train_iterator_and_tokenizer(config, mesh, add_bos, add_eos):
  """ Make train iterator and tokenizer for customized C4 dataset for mlperf gpt3 training."""
  train_ds, eval_ds = _tfds_data_processing_c4_mlperf.get_datasets(
    config=config,
  )
  sp_tokenizer = get_tokenizer(config.tokenizer_path, add_bos, add_eos)
  train_iter, eval_iter = _tfds_data_processing_c4_mlperf.preprocess_dataset(
    config,
    mesh,
    train_ds, eval_ds, sp_tokenizer,
    data_shuffle_seed=config.data_shuffle_seed
  )
  return train_iter, eval_iter, sp_tokenizer

def make_c4_train_iterator_and_tokenizer(config, mesh, add_bos, add_eos):
  """ Make train iterator and tokenizer for C4 dataset"""
  read_config = tfds.ReadConfig(
    shuffle_seed = config.data_shuffle_seed,
  )
  train_ds, eval_ds = _tfds_data_processing.get_datasets(
    config=config,
    read_config = read_config,
  )
  sp_tokenizer = get_tokenizer(config.tokenizer_path, add_bos, add_eos)
  train_iter, _, _ = _tfds_data_processing.preprocess_dataset(
    config,
    mesh,
    train_ds, eval_ds, sp_tokenizer,
    data_shuffle_seed = config.data_shuffle_seed,
  )
  return train_iter, None, sp_tokenizer

def make_grain_train_iterator_and_tokenizer(config, mesh, add_bos, add_eos):
  """ Make train iterator and tokenizer for C4 dataset"""
  train_ds, eval_ds = _grain_data_processing.get_datasets(
    config=config
  )
  sp_tokenizer = get_tokenizer(config.tokenizer_path, add_bos, add_eos)
  train_iter, _, _ = _grain_data_processing.preprocess_dataset(
    config,
    mesh,
    train_ds, eval_ds,
    vocab_path=config.tokenizer_path,
    data_shuffle_seed = config.data_shuffle_seed,
    add_bos = add_bos,
    add_eos = add_eos
  )
  return train_iter, None, sp_tokenizer


def extract_train_skip_step(job_log_dir, step, only_eval=False):
  # Todo need rewrite
    if job_log_dir is None:
        return {}
    model_dir = job_log_dir / "checkpoints"
    if step is not None:
        fill_step = checkpoint_paths.CHECKPOINT_PREFIX + str(step).zfill(checkpoint_paths._STEP_FORMAT_FIXED_LENGTH)
        skip_file_and_step_path = model_dir / fill_step / checkpoint_paths.SKIP_STEP_NAME
    else:
        skip_file_and_step_path = model_dir / checkpoint_paths.SKIP_STEP_NAME
    logging.info(f"model_dir: {model_dir}")
    try:
        with skip_file_and_step_path.open('r') as f:
            meta_dict = json.load(f)
        logging.info(f"Load skip_file_and_step_path: ’{skip_file_and_step_path}‘ Finished.......")
    except:
        logging.info(f"skip_file_and_step_path: ’{skip_file_and_step_path}‘ is not existed.......")
        meta_dict = {}

    if jax.process_index() == 0:
        mode = 'train_break_steps' if not only_eval else 'eval_metric_steps'
        back_meta_dict_path = job_log_dir / mode /f'{meta_dict.get("checkpoint_step", None)}.json'
        with back_meta_dict_path.open('w') as f1:
            json.dump(meta_dict, f1)
    return meta_dict


def extract_pythia_datapath(dataset_path, eval_split):
    if not dataset_path:
      return []
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    logging.info(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
    step_map_path = {}
    eval_pathes = []
    rerank = 0
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        if ".tfrecord" not in blob.name: continue
        try:
            step = int(blob.name.rsplit("pile.tfrecord.b", maxsplit=1)[-1])
        except:
            step = rerank
            rerank += 1
        path = f'gs://{os.path.join(bucket_name, blob.name)}'

        if eval_split in path:
            logging.info(f'eval path: {path}')
            eval_pathes.append(path)
            continue
        step_map_path[step] = path

    sorted_step_path = sorted(step_map_path.items(), key=lambda x: x[0])
    steps, pathes = zip(*sorted_step_path)
    if not isinstance(pathes, list):
        pathes = list(pathes)
    logging.info(f'pathes: {len(pathes)} eval_pathes: {len(eval_pathes)}')
    return pathes, eval_pathes

# pile
def make_pile_train_iterator(config, mesh, add_bos, add_eos):
  train_name = f'{config.dataset_type}.train'
  eval_name = f'{config.dataset_type}.eval'

  train_pathes, eval_pathes = extract_pythia_datapath(config.dataset_path, config.eval_split)
  num_local_devices = jax.local_device_count()
  #  checkpoint_dir = os.path.join(base_output_directory, run_name, "checkpoints", "")
  checkpoint_last_dir = os.path.basename(config.checkpoint_dir)
  # meta_dict = extract_train_skip_step(job_log_dir=checkpoint_last_dir, step=config.training_num_batches_to_skip, only_eval=False)
  meta_dict = {}
  num_batches_to_skip = meta_dict.get('checkpoint_step', config.training_num_batches_to_skip)

  task_features = ['input_ids']

  train_dataloader = _pile_data_processing.PileDatasets(
                            mesh=mesh,
                            name=train_name, 
                            path=train_pathes, 
                            meta_dict=meta_dict,
                            batch_size=int(config.per_device_batch_size * num_local_devices),
                            seq_len=config.max_target_length,
                            repeat=config.epoch,
                            seed=config.data_shuffle_seed,
                            task_features=task_features,
                            shuffle_buffer_size=config.train_shuffle_buffer_size,
                            num_batches_to_skip=num_batches_to_skip,
                            only_eval=False,
                            zero_loss=config.zero_loss,
                            iter_file_nums=config.iter_file_nums,
                            )
  eval_dataloader = None
  if eval_pathes:
    eval_dataloader = _pile_data_processing.PileDatasets(
                            mesh=mesh,
                            name=eval_name, 
                            path=eval_pathes, 
                            meta_dict={},
                            batch_size=int(config.eval_per_device_batch_size * num_local_devices),
                            seq_len=config.max_target_length,
                            repeat=config.epoch,
                            seed=config.data_shuffle_seed,
                            task_features=task_features,
                            shuffle_buffer_size=config.eval_shuffle_buffer_size,
                            num_batches_to_skip=None,
                            only_eval=False,
                            zero_loss=config.zero_loss,
                            iter_file_nums=config.iter_file_nums,
                            )
  return train_dataloader, eval_dataloader, None


class SyntheticDataIterator():
  """Creates a synthetic data iterator for performance testing work"""
  def __init__(self, config, mesh):
    self.mesh = mesh
    self.config = config
    data_pspec = P(*config.data_sharding)
    data_pspec_shardings = jax.tree_map(
        lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
    self.data_generator = jax.jit(SyntheticDataIterator.raw_generate_synthetic_data,
        out_shardings=data_pspec_shardings,
        static_argnums=0)

  def __iter__(self):
    return self

  def __next__(self):
    with self.mesh:
      return self.data_generator(self.config)

  @staticmethod
  def raw_generate_synthetic_data(config):
    """Generates a single batch of syntehtic data"""
    output = {}
    output['inputs'] = jax.numpy.zeros( (config.global_batch_size_to_load, config.max_target_length),
                                       dtype=jax.numpy.int32)
    output['inputs_position'] = jax.numpy.zeros( (config.global_batch_size_to_load, config.max_target_length),
                                                dtype=jax.numpy.int32)
    output['inputs_segmentation'] = jax.numpy.ones( (config.global_batch_size_to_load, config.max_target_length),
                                                   dtype=jax.numpy.int32)
    output['targets'] = jax.numpy.zeros( (config.global_batch_size_to_load, config.max_target_length),
                                        dtype=jax.numpy.int32)
    output['targets_position'] = jax.numpy.zeros( (config.global_batch_size_to_load, config.max_target_length),
                                                 dtype=jax.numpy.int32)
    output['targets_segmentation'] = jax.numpy.ones( (config.global_batch_size_to_load, config.max_target_length),
                                                    dtype=jax.numpy.int32)
    return output


def create_data_iterator_with_tokenizer(config, mesh, add_bos = True, add_eos = True):
  if config.dataset_type == "synthetic":
    return SyntheticDataIterator(config, mesh), None, get_tokenizer(config.tokenizer_path, add_bos, add_eos)
  elif config.dataset_type == "c4":
    return make_c4_train_iterator_and_tokenizer(config, mesh, add_bos, add_eos)
  elif config.dataset_type == "c4-array_record":
    return make_grain_train_iterator_and_tokenizer(config, mesh, add_bos, add_eos)
  elif config.dataset_type == "c4_mlperf":
    print("Overwrite both add_bos and add_eos to False")
    return make_c4_mlperf_train_iterator_and_tokenizer(config, mesh, add_bos=False, add_eos=False)
  elif config.dataset_type == "pile":
    return make_pile_train_iterator(config, mesh, add_bos, add_eos)
  else:
    assert False, "dataset type not implemented"

def get_shaped_batch(config):
  """ Return the shape of the batch - this is what eval_shape would return for the
  output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078."""
  batch_shape = (config.global_batch_size_to_load, config.max_target_length)
  shaped_batch = {}
  shaped_batch['inputs'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['inputs_position'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['inputs_segmentation'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['targets'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['targets_position'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['targets_segmentation'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  return shaped_batch
