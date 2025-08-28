"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

from needle.autograd import compute_gradient_of_variables

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
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
        X = X.reshape(num_images, num_rows * num_cols)

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


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### Average loss over the batch (one sample -> loss)
    left = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    # element-wise multiplication, 每一行只剩下一个非零值，然后fold
    right = ndl.summation(Z * y_one_hot, axes=(1,))
    # average batch gradient
    return ndl.summation((left - right), axes=(0,)) / Z.shape[0]


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    num_examples = X.shape[0]
    num_classes = W2.shape[1]

    for start in range(0, num_examples, batch):
        end = min(start + batch, num_examples)
        X_batch = X[start:end]
        y_batch = y[start:end]
        X_batch = ndl.Tensor(X_batch)
        # y_batch = ndl.Tensor(y_batch)
        # Forward pass
        hidden = ndl.matmul(X_batch, W1)  # (batch, hidden_dim)
        hidden_relu = ndl.relu(hidden)  # ReLU activation
        logits = ndl.matmul(hidden_relu, W2)

        y_one_hot = np.zeros((y_batch.shape[0], num_classes))
        y_one_hot[np.arange(y_batch.size), y_batch] = 1
        y_ = ndl.Tensor(y_one_hot)
        loss = softmax_loss(logits, y_)

        loss.backward()
        # W1 -= W1.grad * lr
        # W2 -= W2.grad * lr
        # W1.grad, W2.grad = None, None
        W1_new_data = W1.numpy() - lr * W1.grad.numpy()
        W2_new_data = W2.numpy() - lr * W2.grad.numpy()

        # 从新的 data 创建新的叶子节点 Tensor
        W1 = ndl.Tensor(W1_new_data)
        W2 = ndl.Tensor(W2_new_data)
        # Backward pass
    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
