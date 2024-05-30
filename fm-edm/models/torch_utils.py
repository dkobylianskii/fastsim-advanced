"""Mix of utility functions specifically for pytorch.
https://github.com/rodem-hep/nu2flows
"""

from functools import partial

import numpy as np
import torch as T
import torch.nn as nn

# An onnx save argument which is for the pass with mask function (makes it slower)
ONNX_SAFE = False


def append_dims(x: T.Tensor, target_dims: int, add_to_front: bool = False) -> T.Tensor:
    """
    Appends dimensions of size 1 to the end or front of a tensor until it has
    target_dims dimensions.

    Parameters
    ----------
    x : T.Tensor
        The input tensor to be reshaped.
    target_dims : int
        The target number of dimensions for the output tensor.
    add_to_front : bool, optional
        If True, dimensions are added to the front of the tensor.
        If False, dimensions are added to the end of the tensor.
        Defaults to False.

    Returns
    -------
    T.Tensor
        The reshaped tensor with target_dims dimensions.

    Raises
    ------
    ValueError
        If the input tensor already has more dimensions than target_dims.

    Examples
    --------
    >>> x = T.tensor([1, 2, 3])
    >>> x.shape
    torch.Size([3])

    >>> append_dims(x, 3)
    tensor([[[1]], [[2]], [[3]]])
    >>> append_dims(x, 3).shape
    torch.Size([3, 1, 1])

    >>> append_dims(x, 3, add_to_front=True)
    tensor([[[[1, 2, 3]]]])
    >>> append_dims(x, 3, add_to_front=True).shape
    torch.Size([1, 1, 3])

    """
    dim_diff = target_dims - x.dim()
    if dim_diff < 0:
        raise ValueError(f"x has more dims ({x.ndim}) than target ({target_dims})")
    if add_to_front:
        return x[(None,) * dim_diff + (...,)]  # x.view(*dim_diff * (1,), *x.shape)
    return x[(...,) + (None,) * dim_diff]  # x.view(*x.shape, *dim_diff * (1,))


def get_act(name: str) -> nn.Module:
    """Return a pytorch activation function given a name."""
    if isinstance(name, partial):
        return name()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "lrlu":
        return nn.LeakyReLU(0.1)
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "selu":
        return nn.SELU()
    if name == "softmax":
        return nn.Softmax()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softmax":
        return nn.Softmax()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError("No activation function with name: ", name)


def get_nrm(name: str, outp_dim: int) -> nn.Module:
    """Return a 1D pytorch normalisation layer given a name and a output size
    Returns None object if name is none."""
    if name == "batch":
        return nn.BatchNorm1d(outp_dim)
    if name == "layer":
        return nn.LayerNorm(outp_dim)
    if name == "none":
        return None
    else:
        raise ValueError("No normalistation with name: ", name)
