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

"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""

from typing import Optional, Union
from etils import epath
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions
import orbax.checkpoint
import grain.python as grain

import max_logging
from multihost_dataloading import MultiHostDataLoadIterator
from flax.training import train_state

def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
    dataset_type: Optional[str] = 'c4'
):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None
  max_logging.log("Creating checkpoint manager...")
  p = epath.Path(checkpoint_dir)

  if dataset_type=='c4-array_record':
    item_names = ('state', 'iter')
  else:
    item_names = ('state',)

  items = {
        "state": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler(use_ocdbt=False)), # lsp
        # "step": orbax.checkpoint.Checkpointer(orbax.checkpoint.ArrayCheckpointHandler()),
    }

  mngr = CheckpointManager(
      p,
      items,
      # item_names = item_names,
      options = CheckpointManagerOptions(
          create=True,
          save_interval_steps=save_interval_steps,
          enable_async_checkpointing=use_async,
      )
  )
  max_logging.log("Checkpoint manager created!")
  return mngr


def load_state_if_possible(checkpoint_manager: CheckpointManager,
                           config,
                           data_iterator: Union[MultiHostDataLoadIterator, None],
                           abstract_unboxed_pre_state: train_state.TrainState):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    load_parameters_from_path: if there is no checkpoint in the checkpoint manager,
      load parameters from a parameter only checkpoint at this path.
    load_full_state_from_path: if there is no checkpoint in the checkpoint manager,
      load full state from a full state checkpoint at this path.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
      matches type against.
    mesh: a physical TPU mesh
    state_mesh_annotation: a PyTree of sharding rules, matching
      abstract_unboxed_pre_state.

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """
  job_dir = epath.Path(config.checkpoint_dir)
  meta_dict = data_iterator.meta_dict
  checkpoint_step = meta_dict.get('checkpoint_step', None)

  if config.load_full_state_from_path:
    checkpoint_dir = epath.Path(config.load_full_state_from_path)
    max_logging.log(f"restoring full state from {config.load_full_state_from_path=}")
    ckptr = orbax.checkpoint.StandardCheckpointer()
    restored = ckptr.restore(checkpoint_dir, args=orbax.checkpoint.args.StandardRestore(abstract_unboxed_pre_state))
    return  {'state': restored}, None

  elif config.load_parameters_from_path:
    checkpoint_dir = epath.Path(config.load_parameters_from_path)
    max_logging.log(f"restoring params from {config.load_parameters_from_path=}")
    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    restore_args = orbax.checkpoint.checkpoint_utils.construct_restore_args(abstract_unboxed_pre_state.params)
    restored = ckptr.restore(checkpoint_dir, item = {'params': abstract_unboxed_pre_state.params}, transforms={},
                            restore_args = {'params': restore_args})
    return None, restored['params']

  elif checkpoint_step is not None:
    max_logging.log(f"restoring params from ’{job_dir}‘ checkpoint_step: {checkpoint_step}")
    checkpoint_dir = job_dir / str(checkpoint_step) / 'state'
    max_logging.log(f"restoring full state from {config.load_full_state_from_path=}")
    ckptr = orbax.checkpoint.StandardCheckpointer()
    restored = ckptr.restore(checkpoint_dir, args=orbax.checkpoint.args.StandardRestore(abstract_unboxed_pre_state))
    return  {'state': restored}, None
    
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None
