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

import max_logging
from etils import epath
import random

SKIP_STEP_NAME = 'skip_file_and_step.json'


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


def record_file_and_step(step, config, train_input):
    save_dir = epath.Path(config.checkpoint_dir)
    save_path = save_dir / str(step) / SKIP_STEP_NAME
    save_newest_path = save_dir / SKIP_STEP_NAME

    if not hasattr(train_input, 'meta_dict'):
        return
    meta_dict = train_input.meta_dict
    meta_dict['checkpoint_step'] = int(step)

    print(f'save_newest_path: {save_newest_path}')
    print(f'save_path: {save_path}')
    print(f'meta_dict: {meta_dict}')
    for k, v in meta_dict.items():
      print(k, type(v))

    # __import__('ipdb').set_trace()
    if jax.process_index() == 0:
      try:
        with save_newest_path.open('w') as f1:
            json.dump(meta_dict, f1)

        with save_path.open('w') as f2:
            json.dump(meta_dict, f2)
      except Exception as error:
        print(f'Write meta dict error: {error}')

    max_logging.log(f'Save skip_file_and_step successful... file_in_data: {meta_dict["file_in_data"]} || step_in_file: {meta_dict["step_in_file"]}')  # XD


def extract_pythia_datapath(dataset_path, eval_split):
    if not dataset_path:
      return []
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    max_logging.log(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
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
            max_logging.log(f'eval path: {path}')
            eval_pathes.append(path)
            continue
        step_map_path[step] = path

    sorted_step_path = sorted(step_map_path.items(), key=lambda x: x[0])
    steps, pathes = zip(*sorted_step_path)
    if not isinstance(pathes, list):
        pathes = list(pathes)
    max_logging.log(f'pathes: {len(pathes)} eval_pathes: {len(eval_pathes)}')
    return pathes, eval_pathes


def extract_v3p5_longdata_files(dataset_path, eval_split=None):
    # random.seed(9876)
    client = storage.Client()
    #v3: us-east1-d -> common_datasets, v4: us-central2-b -> common_datasets_us-central2-b
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    train_files, valid_files = [], []
    train_long_files, train_short_files = [], []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if 'valid' in path:
            valid_files.append(path)
        else:
            if '.long' in path:
                train_long_files.append(path)
            else:
                train_short_files.append(path)
    # file size short：long = 1.5: 1, 为了保证short的token: long = 3: 7, 因此 short 取 (1 / 1.5) * (3 / 7) = 2 / 7
    # k = len(train_short_files) // 1
    short_k = min(3 * len(train_long_files) // 14, len(train_short_files))
    selected_short_files = random.sample(train_short_files, k=short_k)
    train_files = selected_short_files + train_long_files
    max_logging.log(f'selected_short_files: {len(selected_short_files)} train_long_files: {len(train_long_files)}')
    random.shuffle(train_files)
    valid_files = sorted(valid_files)
    return train_files, valid_files


def extract_train_skip_step(job_log_dir, step, only_eval=False):
    if job_log_dir is None:
        return {}
    model_dir = job_log_dir / "checkpoints"
    if step is not None:
        skip_file_and_step_path = model_dir / str(step) / SKIP_STEP_NAME
    else:
        skip_file_and_step_path = model_dir / SKIP_STEP_NAME
    max_logging.log(f"model_dir: {model_dir}")
    try:
        with skip_file_and_step_path.open('r') as f:
            meta_dict = json.load(f)
        max_logging.log(f"Load skip_file_and_step_path: ’{skip_file_and_step_path}‘ Finished.......")
    except:
        max_logging.log(f"skip_file_and_step_path: ’{skip_file_and_step_path}‘ is not existed.......")
        meta_dict = {}

    if jax.process_index() == 0:
        mode = 'train_break_steps' if not only_eval else 'eval_metric_steps'
        back_meta_dict_dir = job_log_dir / mode
        if 'gs:' not in str(back_meta_dict_dir):
          os.makedirs(back_meta_dict_dir, exist_ok=True)
        back_meta_dict_path = back_meta_dict_dir /f'{meta_dict.get("checkpoint_step", None)}.json'
        with back_meta_dict_path.open('w') as f1:
            json.dump(meta_dict, f1)
    return meta_dict

# pile
def make_pile_train_iterator(config, mesh, add_bos, add_eos):
  train_name = f'{config.dataset_type}.train'
  eval_name = f'{config.dataset_type}.eval'

  # train_pathes, eval_pathes = extract_pythia_datapath(config.dataset_path, config.eval_split)
  train_pathes, eval_pathes = extract_v3p5_longdata_files(config.dataset_path, config.eval_split)

  num_local_devices = jax.local_device_count()

  job_dir = epath.Path(config.run_name)
  meta_dict = extract_train_skip_step(job_dir, step=config.training_num_batches_to_skip, only_eval=getattr(config, 'only_eval', False))
  # load_full_state_path
  print(f'meta_dict: {meta_dict}')

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
                            num_batches_to_skip=None,
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
  batch_shape = (config.global_batch_size_to_load, config.max_target_length - 1)
  shaped_batch = {}
  shaped_batch['inputs'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['inputs_position'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['inputs_segmentation'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['targets'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['targets_position'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['targets_segmentation'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  return shaped_batch
