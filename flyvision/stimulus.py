from typing import Dict
from contextlib import contextmanager

import numpy as np
import torch
from torch import Tensor


class Stimulus:
    """Interface to control the neuron-specific stimulus for the network.

    Maps standard video input to the photoreceptors but can map
    input to any other cell type as well, e.g. to do ablation experiments.

    Args:
        n_samples (int): number of samples.
        n_frames (int): number of stimulus frames.a
        ctome (Connectome): instance of a connectome.

    Returns:
        Tensor: stimulus of shape (#samples, #frames, #nodes)

    Example:
        stim = Stimulus(24, 4, 44986, network.ctome)
        stim.add_input(x)
        stim.add_layer_noise("Tm1", mean=0, std=0)

    Note, call .zero(n_samples, n_frames) to change the dimension of the
    stimulus buffer.
    """

    # TODO: samples first
    layer_index: Dict[str, np.ndarray]
    central_nodes_index: Dict[str, int]
    input_index: np.ndarray
    n_frames: int
    n_samples: int
    n_nodes: int
    stimulus: Tensor

    def __init__(self, n_samples, n_frames, ctome, _init=True):
        self.layer_index = {
            node_type: index[:] for node_type, index in ctome.nodes.layer_index.items()
        }
        self.central_nodes_index = dict(
            zip(
                ctome.unique_node_types[:].astype(str),
                ctome.central_nodes_index[:],
            )
        )
        self.input_index = np.array(
            [
                self.layer_index[node_type.decode()]
                for node_type in ctome.input_node_types[:]
            ]
        )
        self.n_samples, self.n_frames, self.n_nodes = (
            n_samples,
            n_frames,
            len(ctome.nodes.type),
        )
        if _init:
            self.zero()

    def zero(
        self,
        n_samples=None,
        n_frames=None,
    ):
        """Resets the stimulus to zeros"""
        self.n_samples = n_samples or self.n_samples
        self.n_frames = n_frames or self.n_frames
        self.stimulus = torch.zeros((self.n_samples, self.n_frames, self.n_nodes))

    def add_input(self, x, start=None, stop=None):
        """Adds input to the photoreceptor indices.

        Args:
            x (Tensor): an input video of shape (#samples, #frames, 1, #hexals).
        """
        shape = x.shape
        if len(shape) != 4:
            raise ValueError(
                "not all dimensions specified. Before, 3 dimensions"
                " were allowed but samples and frames was ambigious"
                " leading to errors."
            )

        if start is not None and stop is not None:
            n_frames = stop - start
        elif start is not None and stop is None:
            n_frames = self.n_frames - start
        elif start is None and stop is not None:
            n_frames = stop
        else:
            n_samples, n_frames = shape[:2]
            if n_frames != self.n_frames or n_samples != self.n_samples:
                self.zero(n_samples, n_frames)
        self.stimulus[:, slice(start, stop), self.input_index] += x.cuda().view(
            n_samples, n_frames, 1, x.shape[-1]
        )

    def add_pre_stim(self, x, start=None, stop=None):
        """Add a constant.

        Args:
            x (float or array): grey value(s). If array must have length stop-start.
            start (int, optional): start index in time. Defaults to None.
            stop (int, optional): stop index in time. Defaults to None.
        """
        if isinstance(x, torch.Tensor) and x.ndim != 0:
            # This allows to have x vary over time.
            self.stimulus[:, slice(start, stop), self.input_index] += x.view(
                1, len(x), 1, 1
            )
        else:
            self.stimulus[:, slice(start, stop), self.input_index] += x

    def add_layer_stim(self, node_type, x):
        """Adds a stimulus to a hexlattice of a specified cell type.

        Args:
            node_type (str): a cell type, e.g. "T4a"
            x (Tensor): an input sequence of shape (#samples, #frames, 1, #hexals).
        """
        self.stimulus[:, :, self.layer_index[node_type]] += x

    def add_central_stim(self, node_type, x):
        """Adds a stimulus to a central neuron of a specified cell type.

        Args:
            node_type (str): a cell type, e.g. "T4a"
            x (Tensor): an input sequence of shape (#samples, #frames).
        """
        self.stimulus[:, :, self.central_nodes_index[node_type]] += x

    def add_layer_noise(self, node_type, mean, std):
        """Adds gaussian noise to a hexlattice of a specified cell type.

        Args:
            node_type (str): a cell type, e.g. "T4a"
            mean (float): mean of the gaussian noise.
            std (float): standard deviation of the gaussian noise.
        """
        noise = (
            torch.randn_like(self.stimulus[:, :, self.layer_index[node_type]]) * std
            + mean
        )
        self.stimulus[:, :, self.layer_index[node_type]] += noise

    def add_central_noise(self, node_type, mean, std):
        """Adds gaussian noise to the central neuron of a specified cell type.

        Args:
            node_type (str): a cell type, e.g. "T4a"
            mean (float): mean of the gaussian noise.
            std (float): standard deviation of the gaussian noise.
        """
        noise = (
            torch.randn_like(self.stimulus[:, :, self.central_nodes_index[node_type]])
            * std
            + mean
        )
        self.stimulus[:, :, self.central_nodes_index[node_type]] += noise

    def suppress_layer(self, node_type, value=-1e3):
        """Adds large negative potential to a hexlattice of a specified cell type.

        Args:
            node_type (str): a cell type, e.g. "T4a"
            value (float): negative input.
        """
        self.stimulus[:, :, self.layer_index[node_type]] += value

    def suppress_center(self, node_type, value=-1e3):
        """Adds large negative potential to the central neuron of a specified cell type.

        Args:
            node_type (str): a cell type, e.g. "T4a"
            value (float): negative input.
        """
        self.stimulus[:, :, self.central_nodes_index[node_type]] += value

    def __call__(self):
        """Returns the stimulus tensor."""
        return self.stimulus

    @contextmanager
    def memory_friendly(self):
        """To remove the stimulus buffer temporarily to save GPU memory.

        TODO: benchmark cost and effect to see if that is good.
        """
        yield
        delattr(self, "stimulus")
