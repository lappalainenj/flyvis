"""Rendering utils"""
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F


# ----- Supporting definitions -------------------------------------------------


def median(x, kernel_size, stride=1, n_chunks=10):
    """Median image filter with reflected padding.

    x (array or tensor): (#samples, #frames, height, width).
                        First and second dimension are optional.
    kernel_size (int): size of the boxes.
    n_chunks (int): chunksize over #samples, #frames to process the data on the
        gpu if it runs out of memory.

    Note: On the gpu it creates a tensor of kernel_size ** 2 * prod(x.shape)
        elements which consumes a lot of memory.
        ~ 14 GB for 50 frames of extent 436, 1024 with kernel_size 13.
        In case of a RuntimeError due to memory the methods processes the data
        in chunks.
    """
    # Get padding so that the resulting tensor is of the same shape.
    p = max(kernel_size - 1, 0)
    p_floor = p // 2
    p_ceil = p - p_floor
    padding = (p_floor, p_ceil, p_floor, p_ceil)

    shape = x.shape

    if not isinstance(x, torch.cuda.FloatTensor):
        # to save time at library import
        from scipy.ndimage import median_filter

        # Process on cpu using scipy.ndimage.median_filter.
        _type = np.array
        if isinstance(x, torch.FloatTensor):
            x = x.numpy()
            _type = torch.FloatTensor
        x = x.reshape(-1, *x.shape[-2:])

        def map_fn(z):
            return median_filter(z, size=kernel_size)

        x = np.concatenate(tuple(map(map_fn, x)), axis=0)
        return _type(x.reshape(shape))

    # Process on gpu.
    try:
        with torch.no_grad():
            x.unsqueeze_(0).unsqueeze_(0) if len(shape) == 2 else x.unsqueeze_(
                0
            ) if len(shape) == 3 else None
            assert len(x.shape) == 4
            _x = F.pad(x, padding, mode="reflect")
            _x = _x.unfold(dimension=2, size=kernel_size, step=stride).unfold(
                dimension=3, size=kernel_size, step=stride
            )
            _x = _x.contiguous().view(shape[:4] + (-1,)).median(dim=-1)[0]
            return _x.view(shape)
    except RuntimeError:
        torch.cuda.empty_cache()
        _x = x.reshape(-1, *x.shape[-2:])
        chunks = torch.chunk(_x, max(n_chunks, 1), dim=0)

        def map_fn(z):
            return median(z, kernel_size, n_chunks=n_chunks - 1)

        _x = torch.cat(tuple(map(map_fn, chunks)), dim=0)
        return _x.view(shape)


def hex_center_coordinates(n_hex_area, img_width, img_height, center=True):

    # Horizontal extent of the grid
    n = np.floor(np.sqrt(n_hex_area / 3)).astype("int")

    dist_h = img_height / (2 * n + 1)
    dist_w = img_width / (2 * n + 1)

    xs = []
    ys = []
    for q in range(-n, n + 1):
        for r in range(max(-n, -n - q), min(n, n - q) + 1):
            xs.append(dist_w * r)
            ys.append(
                dist_h * (q + r / 2)
            )  # either must be negative or origin must be upper
    xs, ys = np.array(xs), np.array(ys)
    if center:
        xs += img_width // 2
        ys += img_height // 2
    return xs, ys, (dist_w, dist_h)


def is_inside_hex(
    x,
    y,
    x_centers,
    y_centers,
    dist_to_edge,
    tilt,
    device="cuda",
    dtype=torch.float16,
):
    """Finds whether the given points are inside the given hexagons.

    Args:
        x, y (tensors): Cartesian coordinates of the points, (n_points).
        x_centers, y_centers (tensors): Cartesian centers of the hexagon, (n_hexagons).
        dist_to_edge (float): Euclidian distance from center to edge of the hexagon.
        tilt (float or tensor): Angle of hexagon counter-clockwise tilt, radians.
    Returns:
        vertices (tensor): Cartesian coordinates of the hexagons' vertices,
            (7, 2, n_hexagons).
        is_inside (boolean tensor): Whether points inside or not,
            (n_points, n_hexagons).
    """

    if not isinstance(tilt, torch.Tensor):
        tilt = torch.tensor(tilt, device=device)

    R = torch.tensor(
        [
            [torch.cos(tilt), -torch.sin(tilt)],
            [torch.sin(tilt), torch.cos(tilt)],
        ],
        dtype=dtype,
        device=device,
    )  # rotation matrix
    pi = torch.tensor(np.pi, device=device, dtype=dtype)
    R60 = torch.tensor(
        [
            [torch.cos(pi / 3), -torch.sin(pi / 3)],
            [torch.sin(pi / 3), torch.cos(pi / 3)],
        ],
        dtype=dtype,
        device=device,
    )  # rotation matrix

    # Generate hexagon vertices
    dist_to_vertex = 2 / np.sqrt(3) * dist_to_edge
    vertices = torch.zeros(7, 2, dtype=dtype, device=device)
    vertices[0, :] = torch.matmul(
        R, torch.tensor([dist_to_vertex, 0], dtype=dtype, device=device)
    )
    for i in range(1, 7):
        vertices[i] = torch.matmul(R60, vertices[i - 1])
    vertices = vertices[:, :, None]
    vertices = torch.cat(
        (
            vertices[:, 0:1, :] + x_centers[None, None, :],
            vertices[:, 1:2, :] + y_centers[None, None, :],
        ),
        dim=1,
    )  # (7, 2, n_hexagons)

    # Generate is_inside output
    is_inside = torch.ones(len(x), len(x_centers), dtype=torch.bool, device=device)
    for i in range(6):
        x1, y1 = vertices[i, :, :]  # x1, y1: (n_hexagons)
        x2, y2 = vertices[i + 1, :, :]
        slope = (y2 - y1) / (x2 - x1)  # (n_hexagons)
        f_center = y1 + slope * (x_centers - x1) - y_centers  # (n_hexagons)
        f_points = (
            y1[None, :] + slope[None, :] * (x[:, None] - x1[None, :]) - y[:, None]
        )  # (n_points, n_hexagons)
        is_inside = torch.logical_and(is_inside, f_center.sign() == f_points.sign())

    return vertices, is_inside  # (7, 2, n_hexagons), (n_points, n_hexagons)


def vsplit(array, out_width, n_splits, center_crop_fraction=0.7):
    """Splits an array into n_splits overlapping splits along the last dimension.

    Args:
        array (np.ndarray, torch.Tensor): array of shape (..., width).
        out_width (int): width of output.
        n_splits (int): number of splits.
        center_crop_fraction (float): array will be cropped centrally in width to
            the fraction center_crop_fraction before being partitioned to capture
            more central content, because some sequences have mostly still
            background on the left and right, this avoids creating motion still
            sequences.

    Returns:
        tuple of n_splits arrays: ((..., out_width), ..., (..., out_width))
    """
    if center_crop_fraction is not None:
        return vsplit(
            vcenter_crop(array, center_crop_fraction),
            out_width,
            n_splits,
            center_crop_fraction=None,
        )

    actual_width = array.shape[-1]
    out_width = int(out_width)

    def take(arr, start, stop):
        if isinstance(arr, np.ndarray):
            return np.take(arr, np.arange(start, stop), axis=-1)[None]
        elif isinstance(arr, torch.Tensor):
            return torch.index_select(arr, dim=-1, index=torch.arange(start, stop))[
                None
            ]

    if n_splits == 1:
        out = (array[None, :],)
    elif n_splits > 1:
        _div = int(out_width / n_splits)
        out = ()
        out_width = max(out_width, int(actual_width / n_splits))
        overlap = np.ceil(
            (out_width * n_splits - actual_width) / (n_splits - 1)
        ).astype(int)
        for i in range(n_splits):
            start = i * out_width - i * overlap
            stop = (i + 1) * out_width - i * overlap
            out += (take(array, start, stop),)
    elif n_splits is None or n_splits == 0:
        return array
    else:
        raise ValueError
    if isinstance(array, np.ndarray):
        return np.concatenate(out, axis=0)
    elif isinstance(array, torch.Tensor):
        return torch.cat(out, dim=0)


def vcenter_crop(array, out_width_ratio):
    def take(arr, start, stop):
        if isinstance(arr, np.ndarray):
            return np.take(arr, np.arange(start, stop), axis=-1)
        elif isinstance(arr, torch.Tensor):
            return torch.index_select(arr, dim=-1, index=torch.arange(start, stop))

    width = array.shape[-1]
    out_width = int(out_width_ratio * width)
    return take(array, (width - out_width) // 2, (width + out_width) // 2)


def cartesian_bars(
    img_height_px,
    img_width_px,
    bar_width_px,
    bar_height_px,
    bar_loc_horizontal_px,
    bar_loc_vertical_px,
    n_bars,
    bar_intensity,
    bg_intensity,
    rotate=0,
):
    """
    All parameters in units of pixels.
    """

    bar_spacing = int(img_width_px / n_bars - bar_width_px)

    height_slice = slice(
        int(bar_loc_vertical_px - bar_height_px / 2),
        int(bar_loc_vertical_px + bar_height_px / 2) + 1,
    )

    img = np.ones([img_height_px, img_width_px]) * bg_intensity

    loc_w = int(bar_loc_horizontal_px - bar_width_px / 2)
    for i in range(n_bars):
        #  Fill background with bars.
        start = max(loc_w + i * bar_width_px + i * bar_spacing, 0)
        width_slice = slice(start, loc_w + (i + 1) * bar_width_px + i * bar_spacing + 1)
        img[height_slice, width_slice] = bar_intensity

    if rotate % 360 != 0:
        img = rotate_image(img, angle=rotate)

    return img


def cartesian_gratings(
    img_height_px,
    img_width_px,
    spatial_period_px,
    grating_intensity,
    bg_intensity,
    grating_height_px=None,
    grating_width_px=None,
    grating_phase_px=0,
    rotate=0,
):
    """
    All parameters in units of pixels.
    """
    # to save time at library import
    from scipy.signal import square

    t = (
        2
        * np.pi
        / (spatial_period_px / img_width_px)
        * (
            np.linspace(-1 / 2, 1 / 2, int(img_width_px))
            - grating_phase_px / img_width_px
        )
    )

    gratings = np.tile(square(t), img_height_px).reshape(img_height_px, img_width_px)
    gratings[gratings == -1] = bg_intensity
    gratings[gratings == 1] = grating_intensity

    if grating_height_px:
        mask = np.ones_like(gratings).astype(bool)

        height_slice = slice(
            int(img_height_px // 2 - grating_height_px / 2),
            int(img_height_px // 2 + grating_height_px / 2) + 1,
        )
        mask[height_slice] = False
        gratings[mask] = 0.5

    if grating_width_px:
        mask = np.ones_like(gratings).astype(bool)

        width_slice = slice(
            int(img_width_px // 2 - grating_width_px / 2),
            int(img_width_px // 2 + grating_width_px / 2) + 1,
        )
        mask[:, width_slice] = False
        gratings[mask] = 0.5

    if rotate % 360 != 0:
        gratings = rotate_image(gratings, angle=rotate)

    return gratings


def cartesian_wn_bars(img_height, img_width, rotate=0):
    """
    All parameters in units of pixels.
    """
    seq = np.zeros([img_height, img_width])

    seq[np.arange(img_height)] = np.random.normal(0.5, 0.25, size=[img_width])[None]

    if rotate % 360 != 0:
        seq = rotate_image(seq, angle=rotate)

    return seq


def rotate_image(img, angle=0):

    h, w = img.shape

    diagonal = int(np.sqrt(h**2 + w**2))

    pad_in_height = (diagonal - h) // 2
    pad_in_width = (diagonal - w) // 2

    img = np.pad(
        img,
        ((pad_in_height, pad_in_height), (pad_in_width, pad_in_width)),
        mode="edge",
    )

    img = Image.fromarray((255 * img).astype("uint8")).rotate(
        angle, Image.BILINEAR, False, None
    )
    img = np.array(img, dtype=float) / 255.0

    padded_h, padded_w = img.shape
    return img[
        pad_in_height : padded_h - pad_in_height,
        pad_in_width : padded_w - pad_in_width,
    ]


def resample(stims, t_stim, dt, dim=0, device="cuda"):
    """Resamples set of stims for given stimulus duration and dt.

    Args:
        stims (tensor): stims of shape (#conditions, #hexals).
        t_stim (float): stimulus duration in seconds.
        dt (float): integration time constant in seconds.

    Returns:
        tensor: stims of shape (#frames, #hexals).
    """
    n_offsets = stims.shape[dim]
    repeats = torch.linspace(0, n_offsets - 1, int(t_stim / dt), device=device).long()
    return torch.index_select(stims, dim, repeats)


def shuffle(stims, randomstate=None):
    """To randomly shuffle stims along the frame dimension.

    Args:
        stims (tensor): of shape (N (optional), #frames, #hexals)
    """
    if len(stims.shape) == 3:
        # assume (smples frames hexals)
        def _shuffle(x):
            return shuffle(x, randomstate)

        return torch.stack(list(map(_shuffle, stims)), dim=0)
    perms = (
        randomstate.permutation(stims.shape[0])
        if randomstate is not None
        else np.random.permutation(stims.shape[0])
    )
    return stims[perms]


def resample_grating(grating, t_stim, dt, temporal_frequency):
    n_frames = int(t_stim / dt)
    t_period = 1 / temporal_frequency
    _grating = resample(grating, t_period, dt)
    _grating = _grating.repeat(np.ceil(n_frames / _grating.shape[0]).astype(int), 1)
    return _grating[:n_frames]


def pad(stim, t_stim, dt, fill=0, mode="end", pad_mode="value"):
    """Pads second to last dimension.

    Args:
        stim (tensor): stimulus (..., n_frames, n_hexals)
    """
    diff = int(t_stim / dt) - stim.shape[-2]
    if diff <= 0:
        return stim

    if mode == "end":
        pad = (0, 0, 0, diff)
    elif mode == "start":
        pad = (0, 0, diff, 0)

    if pad_mode == "value":
        return torch.nn.functional.pad(stim, pad=pad, mode="constant", value=fill)
    elif pad_mode == "continue":
        return repeat_last(stim, -2, diff)
    else:
        return torch.nn.functional.pad(stim, pad=pad, mode=pad_mode)


def repeat_last(stim, dim, n_repeats):
    last = stim.index_select(
        dim, torch.tensor([stim.size(dim) - 1], device=stim.device)
    )
    stim = torch.cat((stim, last.repeat_interleave(n_repeats, dim=dim)), dim=dim)
    return stim
