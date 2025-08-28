import numpy as np

import needle
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """

    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )
        self.is_nd = not isinstance(dataset[0], tuple)

    def __iter__(self):
        self.cur = 0
        # 1. 创建一个扁平的、包含所有索引的一维数组
        ordering = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(ordering)
            self.ordering = np.array_split(
                ordering, range(self.batch_size, len(self.dataset), self.batch_size)
            )

        return self

    def __next__(self):

        if self.cur >= len(self.ordering):
            raise StopIteration

        idx_slice = self.ordering[self.cur]
        # 注意形状应该是 tuple(list,list), 列表推导式会产生list(tuple)
        slce = self.dataset[idx_slice]
        self.cur += 1

        # 检查返回的是否是元组（对应有标签的情况）
        if self.is_nd:
            # 如果不是元组，说明只有数据，没有标签 (NDArrayDataset 的情况)
            # 直接转换并返回单个 Tensor
            return needle.Tensor(slce)
        else:
            return tuple([needle.Tensor(x) for x in slce])
