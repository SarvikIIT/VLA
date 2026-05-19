from copy import deepcopy
import importlib.util
import os

from ml_collections import ConfigDict

from octo.data.oxe.oxe_standardization_transforms import bridge_dataset_transform
from octo.utils.spec import ModuleSpec

_config_path = os.path.join(os.path.dirname(__file__), "../scripts/configs/config.py")
_spec = importlib.util.spec_from_file_location("config", _config_path)
_config_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_module)
get_base_config = _config_module.get_config


def update_config(config: ConfigDict, **kwargs):
    assert isinstance(config, ConfigDict)
    new_config = deepcopy(config)
    _recursive_update(new_config, kwargs)
    return new_config


def _recursive_update(base, overrides):
    """Recursively update a ConfigDict, replacing (not merging) when types differ."""
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], ConfigDict)
            and isinstance(value, dict)
        ):
            _recursive_update(base[key], value)
        else:
            base[key] = value


def get_config():
    base_config = get_base_config("dummy")
    del base_config["dataset_kwargs"]["oxe_kwargs"]
    config = update_config(
        base_config,
        num_steps=2,
        optimizer=dict(
            learning_rate=dict(
                warmup_steps=1,
            ),
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1,
            num_val_batches=2,
        ),
        viz_kwargs=dict(
            eval_batch_size=2,
            trajs_for_metrics=4,
            trajs_for_viz=4,
            samples_per_state=4,
        ),
        log_interval=1,
        eval_interval=2,
        viz_interval=2,
        save_interval=2,
        eval_datasets=None,
        dataset_kwargs={
            "dataset_kwargs_list": [
                {
                    "name": "bridge_dataset",
                    "data_dir": "./tests/debug_dataset",
                    "image_obs_keys": {"primary": "image_0"},
                    "proprio_obs_key": "proprio",
                    "language_key": "language_instruction",
                    "standardize_fn": ModuleSpec.create(bridge_dataset_transform),
                },
            ],
            "frame_transform_kwargs": {
                "resize_size": {"primary": (128, 128)},
                "num_parallel_calls": 4,
            },
            "traj_transform_threads": 1,  # shared between all datasets
            "traj_read_threads": 1,  # shared between all datasets
            "batch_size": 64,
            "sample_weights": None,
            "shuffle_buffer_size": 1000,
            "balance_weights": True,
        },
    )
    return config
