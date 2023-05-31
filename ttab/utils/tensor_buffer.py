# -*- coding: utf-8 -*-
from typing import List, Tuple

import torch


class TensorBuffer(object):
    """Packs multiple tensors into one flat buffer."""

    def __init__(self, tensors: List[torch.Tensor], use_cuda: bool = True) -> None:
        indices: List[int] = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors_len = len(tensors)
        self._tensors_sizes = [x.size() for x in tensors]

        self.buffer = flatten(tensors, use_cuda=use_cuda)  # copies

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(
            self._tensors_sizes[index]
        )

    def __len__(self) -> int:
        return self._tensors_len

    def is_cuda(self) -> bool:
        return self.buffer.is_cuda

    def nelement(self) -> int:
        return self.buffer.nelement()

    def unpack(self, tensors: List[torch.Tensor]) -> None:
        for tensor, entry in zip(tensors, self):
            tensor.data[:] = entry


def flatten(
    tensors: List[torch.Tensor],
    shapes: List[Tuple[int, int]] = None,
    use_cuda: bool = True,
):
    # init and recover the shapes vec.
    pointers: List[int] = [0]
    if shapes is not None:
        for shape in shapes:
            pointers.append(pointers[-1] + shape[1])
    else:
        for tensor in tensors:
            pointers.append(pointers[-1] + tensor.nelement())

    # flattening.
    current_device = tensors[0].device
    target_device = tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu"
    vec = torch.empty(pointers[-1], device=target_device)

    for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
        vec[start_idx:end_idx] = (
            tensor.data.view(-1).to(device=target_device)
            if current_device != target_device
            else tensor.data.view(-1)
        )
    return vec


def unflatten(
    self_tensors: List[torch.Tensor],
    out_tensors: List[torch.Tensor],
    shapes: Tuple[int, int],
):
    pointer: int = 0

    for self_tensor, shape in zip(self_tensors, shapes):
        param_size, nelement = shape
        self_tensor.data[:] = out_tensors[pointer : pointer + nelement].view(param_size)
        pointer += nelement
