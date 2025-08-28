import gzip
import struct
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np


def parse_mnist(image_filename, label_filename):
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, "rb") as f_images, gzip.open(
        label_filename, "rb"
    ) as f_labels:
        s = struct.Struct(">I I I I")  # big endian
        values = s.unpack(f_images.read(s.size))
        assert values[0] == 2051
        num_images, num_rows, num_cols = values[1], values[2], values[3]
        X = np.frombuffer(
            f_images.read(num_images * num_rows * num_cols), dtype=np.uint8
        )
        # H * W * C
        X = X.reshape(num_images, num_rows, num_cols, 1)

        s = struct.Struct(">I I")
        values = s.unpack(f_labels.read(s.size))
        assert values[0] == 2049
        label_num = values[1]
        y = np.frombuffer(f_labels.read(label_num), dtype=np.uint8)
        # --- 满足 Docstring 的最后要求 ---
        # 1. 转换数据类型为 float32
        X = X.astype(np.float32)
        # 2. 归一化数据到 [0.0, 1.0]
        X /= 255.0

    return X, y


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.data, self.labels = parse_mnist(image_filename, label_filename)
        self.transforms = transforms

    # TODO mrk indexing logic
    def __getitem__(self, index) -> object:
        data = self.data[index]
        if self.transforms:
            for t in self.transforms:
                data = t(data)
        return data, self.labels[index]

    def __len__(self) -> int:
        return self.data.shape[0]
