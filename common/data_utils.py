# -*- coding: utf-8 -*-
"""数据集加载工具。

说明：机房网络不通时，torchvision 会自动重试下载；也可手动把数据放到
DATA_ROOT 对应目录下（FashionMNIST/raw、cifar-10-batches-py 等）。
"""
import os
import pickle
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset

DATA_ROOT = os.environ.get("DL_DATA_ROOT", "./data")

FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


class LocalCIFAR10(Dataset):
    """本地 CIFAR-10 读取器，直接从 cifar-10-batches-py 的 pickle 读取。

    为什么不用 torchvision.datasets.CIFAR10：后者会用 MD5 校验数据完整性，
    而从 HF 镜像重建的文件虽内容与官方一致、字节却不同，校验必然失败。
    本类跳过 MD5，只要 pickle 能解析出 data/labels 即可使用。
    """

    def __init__(self, root, train=True, transform=None):
        base = os.path.join(root, "cifar-10-batches-py")
        files = ([f"data_batch_{i}" for i in range(1, 6)] if train
                 else ["test_batch"])
        blocks, labels, names = [], [], []
        for f in files:
            p = os.path.join(base, f)
            if not os.path.exists(p):
                raise FileNotFoundError(f"缺少 {p}，请先运行 fetch_cifar10_fast.py")
            with open(p, "rb") as fo:
                entry = pickle.load(fo, encoding="latin1")
            blocks.append(np.asarray(entry["data"]))
            labels += list(entry["labels"])
            names += list(entry.get("filenames", []))
        # (N, 3072) -> (N, 3, 32, 32) -> (N, 32, 32, 3) 便于 PIL 读取
        self.data = np.vstack(blocks).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = labels
        self.filenames = names
        self.classes = CIFAR10_CLASSES
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = Image.fromarray(self.data[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]


def _random_subset(ds, n, seed=42):
    """随机取子集（而非取前 n 条）。

    官方数据集常按类别排过序（IMDB 前半段全是负例），直接切片会得到
    单一标签的子集，导致准确率虚高到 1.0 而模型其实什么都没学到。
    """
    n = min(n, len(ds))
    if n >= len(ds):
        return ds
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    return Subset(ds, order[:n])


def load_fashion_mnist(batch_size=128, root=DATA_ROOT, limit=None,
                       flatten=False):
    """Fashion-MNIST，28×28 灰度图，10 类。limit 可限制样本数以快速调试。"""
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train = torchvision.datasets.FashionMNIST(root=root, train=True,
                                              transform=tf, download=True)
    test = torchvision.datasets.FashionMNIST(root=root, train=False,
                                             transform=tf, download=True)
    if limit:
        train = _random_subset(train, limit)
        test = _random_subset(test, limit // 5 or 1)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=batch_size, shuffle=False))


def load_cifar10(batch_size=128, root=DATA_ROOT, limit=None, augment=False):
    """CIFAR-10，32×32 彩色图，10 类。augment=True 时启用数据增强。"""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    if augment:
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)])
    else:
        train_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    base = os.path.join(root, "cifar-10-batches-py")
    if os.path.exists(base):
        # 本地已就绪（含从 HF 镜像重建的），跳过 MD5 校验直接读
        train = LocalCIFAR10(root, train=True, transform=train_tf)
        test = LocalCIFAR10(root, train=False, transform=test_tf)
    else:
        train = torchvision.datasets.CIFAR10(root=root, train=True,
                                             transform=train_tf, download=True)
        test = torchvision.datasets.CIFAR10(root=root, train=False,
                                            transform=test_tf, download=True)
    if limit:
        train = _random_subset(train, limit)
        test = _random_subset(test, limit // 5 or 1)
    return (DataLoader(train, batch_size=batch_size, shuffle=True,
                       num_workers=0),
            DataLoader(test, batch_size=batch_size, shuffle=False,
                       num_workers=0))


def show_samples(loader, class_names, n=12, save_path=None, show=False):
    """可视化一批样本。"""
    import matplotlib.pyplot as plt
    x, y = next(iter(loader))
    n = min(n, x.size(0))
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.8 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i in range(n):
        img = x[i]
        if img.shape[0] == 1:
            axes[i].imshow(img.squeeze(0), cmap="gray")
        else:
            axes[i].imshow(img.permute(1, 2, 0) * 0.5 + 0.5)
        axes[i].set_title(class_names[y[i]], fontsize=9)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"已保存: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def subset_loader(dataset, n, batch_size=64, shuffle=True):
    return DataLoader(Subset(dataset, range(min(n, len(dataset)))),
                      batch_size=batch_size, shuffle=shuffle)
