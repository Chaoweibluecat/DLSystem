import sys

sys.path.append("../python")
sys.path.append("./python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    l1 = nn.Linear(dim, hidden_dim)
    # batch_norm , feature作为dim
    norm1 = norm(hidden_dim)
    relu1 = nn.ReLU()
    drop_out = nn.Dropout(drop_prob)
    # Residual一般要求输出维度和输出一致
    l2 = nn.Linear(hidden_dim, dim)
    norm2 = norm(dim)
    compose = nn.Sequential(l1, norm1, relu1, drop_out, l2, norm2)
    return nn.Sequential(nn.Residual(compose), nn.ReLU())


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    fl = nn.Flatten()
    l1 = nn.Linear(dim, hidden_dim)
    relu1 = nn.ReLU()
    res_blocks = [
        ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
        for _ in range(num_blocks)
    ]
    l2 = nn.Linear(hidden_dim, num_classes)
    return nn.Sequential(fl, l1, relu1, *res_blocks, l2)


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    err_count = 0
    loss_sum = 0
    if opt:
        model.train()
    else:
        model.eval()
    for batch in dataloader:
        x = batch[0]
        y = batch[1]
        logits = model(x)
        predict = np.argmax(logits.numpy(), axis=1)
        # argsmax返回预测值向量, !=y 操作 得到错误正确的向量
        err_count += np.sum(predict != y.numpy())
        loss = nn.SoftmaxLoss()(logits, y)
        loss_sum += loss.numpy() * y.numpy().shape[0]
        if opt is not None:
            loss.backward()
            opt.step()
            opt.reset_grad()
    return err_count / len(dataloader.dataset), loss_sum / len(dataloader.dataset)


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    net = MLPResNet(784, hidden_dim)
    # opt有状态 当心重复创建
    opt = optimizer(net.parameters(), lr=lr, weight_decay=weight_decay)
    ds = ndl.data.MNISTDataset(
        data_dir + "/train-images-idx3-ubyte.gz",
        data_dir + "/train-labels-idx1-ubyte.gz",
    )
    dl = ndl.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    for e in range(epochs):
        err, loss = epoch(dl, net, opt)
        print(f"Epoch {e}: err {err} loss {loss}")
    net.eval()
    dl_test = ndl.data.DataLoader(
        ndl.data.MNISTDataset(
            data_dir + "/t10k-images-idx3-ubyte.gz",
            data_dir + "/t10k-labels-idx1-ubyte.gz",
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    test_err, test_loss = epoch(dl_test, net)
    return err, loss, test_err, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="./data")
