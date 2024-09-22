"""Stimulus response functions with caching using xarray."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr

import flyvision
from flyvision.analysis import optimal_stimuli
from flyvision.datasets.datasets import StimulusDataset
from flyvision.datasets.dots import CentralImpulses, SpatialImpulses
from flyvision.datasets.flashes import Flashes
from flyvision.datasets.moving_bar import MovingBar, MovingEdge
from flyvision.datasets.sintel import AugmentedSintel

# --------------------- Helper Function ---------------------


@dataclass
class Result:
    stimuli: np.ndarray
    responses: np.ndarray
    arg_df: pd.DataFrame


def compute_responses(
    network: "flyvision.CheckpointedNetwork",
    dataset_class: type,
    dataset_config: Dict,
    batch_size: int,
    t_pre: float,
    t_fade_in: float,
    cell_index: Optional[np.ndarray | str] = "central",
) -> xr.Dataset:
    """Compute responses and return.

    This function is compatible with joblib caching.
    """
    # Reconstruct the network
    network.recover()
    network = network.network  # type: flyvision.Network

    # Initialize dataset
    dataset = dataset_class(**dataset_config)

    if cell_index == "central":
        # Prepare central cells index
        cell_index = network.connectome.central_cells_index[:]

    stimuli_array = []
    responses_array = []

    # Compute responses
    for _, (stimulus, responses) in enumerate(
        network.stimulus_response(
            dataset,
            dataset.dt,
            t_pre=t_pre,
            t_fade_in=t_fade_in,
            batch_size=batch_size,
        )
    ):
        if cell_index is not None:
            responses = np.take(responses, cell_index, axis=-1)

        stimuli_array.append(stimulus)
        responses_array.append(responses)

    stimuli_array = np.concatenate(stimuli_array, axis=0)
    responses_array = np.concatenate(responses_array, axis=0)[None]

    return xr.Dataset(
        {
            'stimulus': (
                [
                    'sample',
                    'frame',
                    'channel',
                    'hex_pixel',
                ],
                stimuli_array,
            ),
            'responses': (
                [
                    'network_id',
                    'sample',
                    'frame',
                    'neuron',
                ],
                responses_array,
            ),
        },
        coords={
            'sample': np.arange(len(dataset)),
            **{
                col: ('sample', dataset.arg_df[col].values)
                for col in dataset.arg_df.columns
            },
        },
    )


def generic_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.Ensemble"],
    dataset,
    dataset_config: Dict,
    default_dataset_cls: type,
    t_pre: float,
    t_fade_in: float,
    batch_size: int,
    cell_index: Optional[np.ndarray | str] = "central",
) -> xr.Dataset:
    """Return responses for a given dataset as an xarray Dataset."""
    # Handle both single and multiple NetworkViews
    if isinstance(network_view_or_ensemble, flyvision.NetworkView):
        network_views = [network_view_or_ensemble]
    else:
        network_views = list(network_view_or_ensemble.values())

    # Prepare dataset class
    dataset_class = default_dataset_cls if dataset is None else type(dataset)
    dataset_config = dataset_config if dataset is None else dataset.config.to_dict()

    # Prepare list to collect datasets
    results = []
    checkpoints = []

    def handle_network(idx, network_view, network=None):
        # Pass initialized network over to next network view to avoid
        # reinitializing the network
        checkpointed_network = network_view.network(
            checkpoint="best", network=network, lazy=True
        )

        # use the cache from this network_view
        cached_compute_responses_fn = network_view.memory.cache(
            compute_responses, ignore=['batch_size']
        )
        call_in_cache = cached_compute_responses_fn.check_call_in_cache(
            checkpointed_network,
            dataset_class,
            dataset_config,
            batch_size,
            t_pre,
            t_fade_in,
            cell_index,
        )

        if call_in_cache:
            # don't initialize the network when the call is in cache
            # print('call in cache')
            pass
        elif network is None and checkpointed_network.network is None:
            # initialize the network when the call is not in cache
            # and the network is not passed from the previous network view
            # print('call not in cache, init network')
            checkpointed_network.init()

        # Call the cached compute_responses function
        results.append(
            cached_compute_responses_fn(
                checkpointed_network,
                dataset_class,
                dataset_config,
                batch_size,
                t_pre,
                t_fade_in,
            )  # type: xr.Dataset
        )
        checkpoints.append(checkpointed_network.checkpoint)
        return checkpointed_network.network

    network = handle_network(0, network_views[0], None)

    for idx, network_view in enumerate(network_views[1:], 1):
        network = handle_network(idx, network_view, network)

    if len(results) > 1:
        # TODO: as long as the concatenation is not lazy, this pattern might not be
        # the best way to handle the results. See also https://github.com/pydata/xarray/issues/4628.
        # TODO: lazy dataset loading could be done by caching this result instead, would
        # require network view or ensemble to be pickable
        result = xr.concat(
            results,
            dim='network_id',
            data_vars='minimal',
            coords='minimal',
            # otherwise repeates stimulus across network_id dim
            compat='override',
        )

    result.coords.update({
        'frame': np.arange(result['stimulus'].shape[1]),
        'channel': np.arange(result['stimulus'].shape[2]),
        'hex_pixel': np.arange(result['stimulus'].shape[3]),
        'neuron': np.arange(result['responses'].shape[3]),
    })

    # Create xarray Dataset with time coordinate
    cell_types = network_views[0].connectome.nodes.type[:].astype(str)
    u_coords = network_views[0].connectome.nodes.u[:]
    v_coords = network_views[0].connectome.nodes.v[:]
    u_in, v_in = np.array([
        (u_coords[i], v_coords[i])
        for i, cell_type in enumerate(cell_types)
        if cell_type == "R1"
    ]).T

    if cell_index == "central":
        # Prepare central cells index
        cell_index = network_views[0].connectome.central_cells_index[:]

    if cell_index is not None:
        cell_types = cell_types[cell_index]
        u_coords = u_coords[cell_index]
        v_coords = v_coords[cell_index]

    # Assuming dataset.dt is available
    dt = dataset_config.get('dt')
    t_pre = dataset_config.get('t_pre', 0.0)
    n_frames = result.frame.size
    time = np.arange(n_frames).astype(float) * dt - t_pre

    # Add relevant coordinates
    result.coords.update({
        'time': ('frame', time),
        'cell_type': ('neuron', cell_types),
        'u': ('neuron', u_coords),
        'v': ('neuron', v_coords),
        'u_in': ('hex_pixel', u_in),
        'v_in': ('hex_pixel', v_in),
        'checkpoints': ('network_id', checkpoints),
    })
    result.attrs.update({
        'config': dataset_config,
        'network_config': network.config.to_dict(),
    })

    return result


# --------------------- Flash Responses ---------------------


# TODO: with network_view pickable, could mem cache this directly.
def flash_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.Ensemble"],
    dataset: Optional[Flashes] = None,
    radius=(-1, 6),
    dt=1 / 200,
    batch_size=4,
) -> xr.Dataset:
    default_dataset_config = {
        'dynamic_range': [0, 1],
        't_stim': 1,
        't_pre': 1.0,
        'dt': dt,
        'radius': radius,
        'alternations': (0, 1, 0),
    }
    return generic_responses(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        Flashes,
        t_pre=1.0,
        t_fade_in=0.0,
        batch_size=batch_size,
    )


# --------------------- Moving Edge Responses ---------------------


def moving_edge_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.Ensemble"],
    dataset: Optional[MovingEdge] = None,
    speeds=(2.4, 4.8, 9.7, 13, 19, 25),
    offsets=(-10, 11),
    dt=1 / 200,
    batch_size=4,
) -> xr.Dataset:
    default_dataset_config = {
        'offsets': offsets,
        'intensities': [0, 1],
        'speeds': speeds,
        'height': 80,
        'post_pad_mode': "continue",
        'dt': dt,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        't_pre': 1.0,
        't_post': 1.0,
    }
    return generic_responses(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        MovingEdge,
        t_pre=1.0,
        t_fade_in=0.0,
        batch_size=batch_size,
    )


# --------------------- Moving Bar Responses ---------------------


def moving_bar_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.Ensemble"],
    dataset: Optional[MovingBar] = None,
    dt=1 / 200,
    batch_size=4,
) -> xr.Dataset:
    default_dataset_config = {
        'widths': [1, 2, 4],
        'offsets': (-10, 11),
        'intensities': [0, 1],
        'speeds': [2.4, 4.8, 9.7, 13, 19, 25],
        'height': 9,
        'post_pad_mode': "continue",
        'dt': dt,
        't_pre': 1.0,
        't_post': 1.0,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
    }
    return generic_responses(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        MovingBar,
        t_pre=1.0,
        t_fade_in=0.0,
        batch_size=batch_size,
    )


# --------------------- Naturalistic Stimuli Responses ---------------------


def naturalistic_stimuli_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.Ensemble"],
    dataset: Optional[AugmentedSintel] = None,
    dt=1 / 100,
    batch_size=4,
) -> xr.Dataset:
    default_dataset_config = {
        'tasks': ["flow"],
        'interpolate': False,
        'boxfilter': {'extent': 15, 'kernel_size': 13},
        'temporal_split': True,
        'dt': dt,
    }
    return generic_responses(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        AugmentedSintel,
        t_pre=0.0,
        t_fade_in=2.0,
        batch_size=batch_size,
    )


# --------------------- Central Impulses Responses ---------------------


def central_impulses_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.Ensemble"],
    dataset: Optional[CentralImpulses] = None,
    intensity=1,
    bg_intensity=0.5,
    impulse_durations=(5e-3, 20e-3, 50e-3, 100e-3, 200e-3, 300e-3),
    dt=1 / 200,
    batch_size=4,
) -> xr.Dataset:
    default_dataset_config = {
        'impulse_durations': impulse_durations,
        'dot_column_radius': 0,
        'bg_intensity': bg_intensity,
        't_stim': 2,
        'dt': dt,
        'n_ommatidia': 721,
        't_pre': 1.0,
        't_post': 0,
        'intensity': intensity,
        'mode': "impulse",
        'device': "cuda" if torch.cuda.is_available() else "cpu",
    }
    return generic_responses(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        CentralImpulses,
        t_pre=4.0,
        t_fade_in=0.0,
        batch_size=batch_size,
    )


# --------------------- Spatial Impulses Responses ---------------------


def spatial_impulses_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.Ensemble"],
    dataset: Optional[SpatialImpulses] = None,
    intensity=1,
    bg_intensity=0.5,
    impulse_durations=(5e-3, 20e-3),
    max_extent=4,
    dt=1 / 200,
    batch_size=4,
) -> xr.Dataset:
    default_dataset_config = {
        'impulse_durations': impulse_durations,
        'max_extent': max_extent,
        'dot_column_radius': 0,
        'bg_intensity': bg_intensity,
        't_stim': 2,
        'dt': dt,
        'n_ommatidia': 721,
        't_pre': 1.0,
        't_post': 0,
        'intensity': intensity,
        'mode': "impulse",
        'device': "cuda" if torch.cuda.is_available() else "cpu",
    }
    return generic_responses(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        SpatialImpulses,
        t_pre=4.0,
        t_fade_in=0.0,
        batch_size=batch_size,
    )


# --------------------- Optimal Stimulus Responses ---------------------


def compute_optimal_stimulus_responses(
    network: "flyvision.CheckpointedNetwork",
    cell_type: str,
    dataset_config: Dict,
    dataset_class: type = AugmentedSintel,
) -> xr.Dataset:
    """Compute optimal stimuli responses and return as xarray Dataset.

    This function is compatible with joblib caching.
    """
    # Create a dummy NetworkView for FindOptimalStimuli
    network_view = flyvision.NetworkView(network_dir=network.name)

    # Prepare dataset configuration
    stimuli_dataset = dataset_class(**dataset_config)
    findoptstim = optimal_stimuli.FindOptimalStimuli(
        network_view, stimuli=stimuli_dataset
    )
    return findoptstim.regularized_optimal_stimuli(cell_type)


def optimal_stimulus_responses(
    network_view: "flyvision.NetworkView",
    cell_type: str,
    dataset: Optional[StimulusDataset] = AugmentedSintel,
    dt=1 / 100,
) -> xr.Dataset:
    """Return optimal stimuli responses as xarray Dataset."""

    # Prepare dataset configuration
    default_dataset_config = {
        'tasks': ["flow"],
        'interpolate': False,
        'boxfilter': {'extent': 15, 'kernel_size': 13},
        'temporal_split': True,
        'dt': dt,
    }

    # Call the cached helper function
    return network_view.memory.cache(compute_optimal_stimulus_responses)(
        network_view.network(checkpoint="best", lazy=True),
        cell_type,
        default_dataset_config,
        dataset,
    )


if __name__ == '__main__':
    # Example usage
    import time

    nv = flyvision.NetworkView("flow/0000/000")
    start = time.time()
    # ds = nv.naturalistic_stimuli_responses()
    # print(ds)
    ds = optimal_stimulus_responses(nv, "T4c")
    print(ds)
    print(f"Elapsed time: {time.time() - start:.2f} seconds")
