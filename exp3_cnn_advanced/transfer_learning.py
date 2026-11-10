# -*- coding: utf-8 -*-
"""实验三 任务3~4：预训练 ResNet18 的特征提取与迁移学习微调。

自定义小数据集：从 CIFAR-10 中抽取 cat / dog 两类，构造"猫狗二分类"数据集
（无需额外下载大模型数据集，机房 CPU 亦可训练）。

三种方式对比：
    A 从头训练（随机初始化 ResNet18）
    B 特征提取（冻结全部卷积层，只训练分类头）
    C 微调（解冻 layer3/layer4 与 fc，小学习率训练）

运行：
    python exp3_cnn_advanced/transfer_learning.py --epochs 5 --per-class 1500
"""
import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import (set_seed, fit, plot_curves, plot_confusion,  # noqa: E402
                          save_comparison_table, count_parameters)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp3_cnn_advanced", "results")
CLASSES = ["cat", "dog"]


class CatsDogsDataset(Dataset):
    """从 CIFAR-10 抽取 cat(3) / dog(5) 构造二分类数据集。"""

    def __init__(self, torchvision_ds, keep=(3, 5), per_class=None):
        self.data, self.labels = [], []
        counts = {c: 0 for c in keep}
        for img, y in torchvision_ds:
            if y in keep and counts[y] < (per_class or 10 ** 9):
                self.data.append(img)
                self.labels.append(0 if y == keep[0] else 1)
                counts[y] += 1
        print(f"   自定义数据集构建完成：{dict(counts)}，共 {len(self.data)} 张")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]


def build_loaders(per_class=None, batch_size=32, img_size=96):
    import torchvision
    base_tf = T.Compose([T.ToTensor(),
                         T.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616))])
    tr = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, "data"),
                                      train=True, transform=base_tf,
                                      download=True)
    te = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, "data"),
                                      train=False, transform=base_tf,
                                      download=True)
    tr_ds = CatsDogsDataset(tr, per_class=per_class)
    te_ds = CatsDogsDataset(te, per_class=max(1, (per_class or 500) // 5))

    # ResNet18 输入建议 ≥96×96，这里上采样后再归一化
    resize = T.Compose([T.Resize((img_size, img_size))])
    class Wrapped(Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            x, y = self.ds[i]
            return resize(x), y

    return (DataLoader(Wrapped(tr_ds), batch_size=batch_size, shuffle=True),
            DataLoader(Wrapped(te_ds), batch_size=batch_size, shuffle=False))


def build_resnet18(pretrained=True, num_classes=2):
    """加载 ResNet18；无网络时自动降级为随机初始化。"""
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        print("   ✓ 已加载 ImageNet 预训练权重")
    except Exception as e:
        print(f"   ! 预训练权重下载失败（{type(e).__name__}），"
              f"改用随机初始化（不影响流程演示）")
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_all_but_fc(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_last_blocks(model):
    """微调：解冻最后两个残差块（layer3、layer4）与 fc。

    指导书原文是"解冻 ResNet 最后两个残差块"。底层 layer1/layer2 学的是
    边缘、纹理这类通用特征，几乎所有视觉任务都用得上，动了反而有害；
    layer3/layer4 才是跟具体类别相关的语义特征，需要往目标域上偏移。
    """
    freeze_all_but_fc(model)
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 60)
    print("构建自定义数据集（cat / dog 二分类）")
    train_loader, test_loader = build_loaders(per_class=args.per_class,
                                              batch_size=args.batch)

    results, rows = {}, []
    # A 从头训练
    print("=" * 60)
    print("方式 A：从头训练（随机初始化）")
    set_seed(42)
    m = build_resnet18(pretrained=not args.no_pretrained and False)
    h = fit(m, train_loader, test_loader, epochs=args.epochs, lr=1e-3,
            optimizer_name="adam", tag="resnet_scratch", verbose=False)
    print(f"   可训练参数={count_parameters(m):,}  best_acc={h['best_acc']:.4f}")
    results["从头训练"] = h["test_acc"]
    rows.append(["从头训练", f"{count_parameters(m):,}", f"{h['best_acc']:.4f}",
                 f"{h['seconds']:.0f}"])

    # B 特征提取
    print("=" * 60)
    print("方式 B：特征提取（冻结卷积层，仅训练分类头）")
    set_seed(42)
    m = build_resnet18(pretrained=not args.no_pretrained)
    freeze_all_but_fc(m)
    h = fit(m, train_loader, test_loader, epochs=args.epochs, lr=1e-3,
            optimizer_name="adam", tag="resnet_feature", verbose=False)
    print(f"   可训练参数={count_parameters(m):,}  best_acc={h['best_acc']:.4f}")
    results["特征提取"] = h["test_acc"]
    rows.append(["特征提取", f"{count_parameters(m):,}", f"{h['best_acc']:.4f}",
                 f"{h['seconds']:.0f}"])

    # C 微调
    print("=" * 60)
    print("方式 C：微调（解冻 layer3 + layer4 + fc，学习率 1e-4）")
    set_seed(42)
    m = build_resnet18(pretrained=not args.no_pretrained)
    unfreeze_last_blocks(m)
    h = fit(m, train_loader, test_loader, epochs=args.epochs, lr=1e-4,
            optimizer_name="adam", tag="resnet_finetune", verbose=False)
    print(f"   可训练参数={count_parameters(m):,}  best_acc={h['best_acc']:.4f}")
    results["微调"] = h["test_acc"]
    rows.append(["微调(layer3+layer4)", f"{count_parameters(m):,}",
                 f"{h['best_acc']:.4f}", f"{h['seconds']:.0f}"])
    plot_confusion(m, test_loader, CLASSES,
                   save_path=os.path.join(OUT, "exp3_transfer_confusion.png"),
                   title="实验三 迁移学习混淆矩阵（微调）", show=False)

    save_comparison_table(rows, ["训练方式", "可训练参数量", "最佳测试准确率",
                                 "用时(s)"],
                          os.path.join(OUT, "exp3_table_transfer.png"),
                          "实验三 迁移学习三种方式对比")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, acc in results.items():
        ax.plot(range(1, len(acc) + 1), acc, "o-", label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("测试准确率")
    ax.set_title("实验三 从头训练 / 特征提取 / 微调 对比")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp3_transfer_curves.png"), dpi=150)
    plt.close(fig)
    print(f"已保存: {OUT}/exp3_transfer_curves.png")


if __name__ == "__main__":
    main()
