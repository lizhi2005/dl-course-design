# -*- coding: utf-8 -*-
"""综合应用一：自定义图像分类数据集构建。

两种使用方式：
    1) 自备数据：把自己的图片按类别放入 data/app1/<类别名>/*.jpg
       例如 data/app1/cat/001.jpg、data/app1/dog/001.jpg
    2) 演示数据：运行本脚本，从 CIFAR-10 抽取若干类生成示例数据集

运行：
    python app1_image_classification/dataset.py --build-demo
    python app1_image_classification/dataset.py --stat
"""
import os
import sys
import argparse

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data", "app1")

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = 128          # CPU 友好：128×128


def build_demo_dataset(classes=("airplane", "automobile", "bird", "cat"),
                       per_class=300):
    """从 CIFAR-10 抽取若干类，存成图片文件夹，作为自定义数据集演示。"""
    cifar = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, "data"),
                                         train=True, download=True)
    idx_of = {c: i for i, c in enumerate(cifar.classes)}
    counts = {c: 0 for c in classes}
    for c in classes:
        os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)
    for img, y in cifar:
        for c in classes:
            if y == idx_of[c] and counts[c] < per_class:
                img.save(os.path.join(DATA_DIR, c, f"{counts[c]:04d}.png"))
                counts[c] += 1
                break
        if all(v >= per_class for v in counts.values()):
            break
    print(f"演示数据集已生成于 {DATA_DIR}：{counts}")
    return counts


def stat():
    """统计数据集类别分布。"""
    if not os.path.exists(DATA_DIR):
        print(f"数据集目录不存在：{DATA_DIR}")
        print("请先放入图片，或运行 --build-demo 生成演示数据")
        return None
    rows = []
    for cls in sorted(os.listdir(DATA_DIR)):
        p = os.path.join(DATA_DIR, cls)
        if os.path.isdir(p):
            n = len([f for f in os.listdir(p)
                     if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))])
            rows.append((cls, n))
    total = sum(n for _, n in rows)
    print(f"数据集：{DATA_DIR}   共 {len(rows)} 类，{total} 张")
    for cls, n in rows:
        print(f"   {cls:<15} {n:>5} 张  ({n/max(total,1)*100:.1f}%)")
    return rows


def get_transforms(train=True):
    """数据预处理与增强。"""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])


def get_loaders(batch_size=32, seed=42, ratios=(0.7, 0.15, 0.15)):
    """构建训练/验证/测试三个 DataLoader。

    划分比例为 7:1.5:1.5，这是指导书里写死的（训练 70%/验证 15%/测试 15%）。

    有个容易踩的点：数据增强只能加在训练集上。验证集和测试集必须只做
    「缩放 + 归一化」，否则同一张图每次评估时被随机翻转/裁剪成不同的样子，
    不同轮次之间的准确率就没有可比性了。所以这里用两份 dataset 对象，
    训练那份带增强，评估那份不带，再按同一组索引取子集。
    """
    from torchvision.datasets import ImageFolder
    from torch.utils.data import Subset
    full = ImageFolder(DATA_DIR, transform=get_transforms(train=True))
    n = len(full)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    n_te = n - n_tr - n_va
    g = torch.Generator().manual_seed(seed)
    tr, va, te = random_split(full, [n_tr, n_va, n_te], generator=g)

    eval_full = ImageFolder(DATA_DIR, transform=get_transforms(train=False))
    va = Subset(eval_full, va.indices)
    te = Subset(eval_full, te.indices)

    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size, shuffle=False),
            DataLoader(te, batch_size=batch_size, shuffle=False),
            full.classes)


def demo_single_image_transform(path):
    """单张图片的推理预处理。"""
    img = Image.open(path).convert("RGB")
    return get_transforms(train=False)(img).unsqueeze(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-demo", action="store_true")
    ap.add_argument("--stat", action="store_true")
    a = ap.parse_args()
    if a.build_demo:
        build_demo_dataset()
    if a.stat or not (a.build_demo):
        stat()
