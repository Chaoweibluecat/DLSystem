import numpy as np


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if not flip_img:
            return img
        return np.flip(img, axis=1)


# todo mark 多维切片, pad
class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """
        对图像进行零填充，然后随机裁剪。

        参数:
             img: H x W x C 格式的图像 NDArray

        返回:
            H x W x C 格式的裁剪后图像 NDArray

        注意: 根据下面指定的 shift_x 和 shift_y 来生成偏移后的图像
        """
        # 获取原始图像的高度和宽度
        h, w, _ = img.shape
        padded = np.pad(
            img,
            ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        # 生成一个在 [-padding, padding] 范围内的随机整数作为偏移量
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )

        # 计算裁剪的起始坐标
        # 加上 padding 是因为坐标系的原点在填充后图像的左上角
        start_x = self.padding + shift_x
        start_y = self.padding + shift_y

        # 使用NumPy的切片功能进行裁剪. [6, 8, 9]
        cropped_img = padded[start_x : start_x + h, start_y : start_y + w, :]

        return cropped_img
