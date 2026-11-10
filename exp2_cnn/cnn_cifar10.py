# -*- coding: utf-8 -*-
"""实验二 任务2~4：构建 CNN 完成 CIFAR-10 分类 + 卷积核/特征图可视化 + 架构对比。

结构（参考《实验指导》）：
    Conv(3→32,3×3,pad=1) → ReLU → MaxPool(2)
    Conv(32→64,3×3,pad=1) → ReLU → MaxPool(2)
    Flatten → FC(64*8*8→256) → ReLU → FC(256→10)

运行：
    python exp2_cnn/cnn_cifar10.py --epochs 10
    python exp2_cnn/cnn_cifar10.py --epochs 1 --limit 10000   # 快速调试
"""
import os
import sys
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data_utils import load_cifar10, CIFAR10_CLASSES, show_samples  # noqa: E402
from common.utils import (set_seed, fit, plot_curves, plot_confusion,  # noqa: E402
                          save_comparison_table, count_parameters)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp2_cnn", "results")


class SimpleCNN(nn.Module):
    """基础 CNN：两层卷积 + 两层全连接，可通过 width/kernels/depth 调整。"""

    def __init__(self, num_classes=10, width=32, kernels=(3, 3), depth=2,
                 use_bn=False):
        super().__init__()
        layers, in_c = [], 3
        for i in range(depth):
            out_c = width * (2 ** i)
            layers += [nn.Conv2d(in_c, out_c, kernels[i], padding=kernels[i] // 2)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            in_c = out_c
        self.features = nn.Sequential(*layers)
        # CIFAR-10 输入 32×32，经过 depth 次 /2 池化
        size = 32 // (2 ** depth)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_c * size * size, 256), nn.ReLU(),
            nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def train_and_report(model, train_loader, test_loader, epochs, tag,
                     lr=1e-3, save=True):
    set_seed(42)
    history = fit(model, train_loader, test_loader, epochs=epochs, lr=lr,
                  optimizer_name="adam", tag=tag, verbose=False)
    print(f"  [{tag}] 参数量={count_parameters(model):,}  "
          f"最佳测试准确率={history['best_acc']:.4f}  "
          f"用时={history['seconds']:.1f}s")
    if save:
        os.makedirs(OUT, exist_ok=True)
        plot_curves(history, save_path=os.path.join(OUT, f"{tag}_curves.png"),
                    title=f"{tag} 训练曲线", show=False)
    return history


def visualize_kernels(model, save_path=None):
    """可视化第一个卷积层的卷积核权重。"""
    conv1 = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv1 = m
            break
    if conv1 is None:
        return
    w = conv1.weight.detach().clone()          # (C_out, C_in, k, k)
    w = (w - w.mean()) / (w.std() + 1e-8)
    n = min(32, w.size(0))
    cols, rows = 8, (n + 7) // 8
    fig, axes = plt.subplots(rows, cols, figsize=(1.3 * cols, 1.3 * rows))
    axes = axes.flatten()
    for i in range(n):
        img = w[i].permute(1, 2, 0).numpy()
        if img.shape[2] == 1:
            axes[i].imshow(img[:, :, 0], cmap="gray")
        else:
            axes[i].imshow((img - img.min()) / (img.max() - img.min() + 1e-8))
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("第一层卷积核可视化（已归一化）")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130)
    print(f"已保存: {save_path}")
    plt.close(fig)


def visualize_feature_maps(model, loader, save_path=None, n=8):
    """可视化第一个卷积层输出的特征图。"""
    model.eval()
    x, _ = next(iter(loader))
    with torch.no_grad():
        conv1 = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                conv1 = m
                break
        feat = torch.relu(conv1(x[:1]))        # (1, C, H, W)
    n = min(n, feat.size(1))
    cols, rows = 4, (n + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows))
    axes = axes.flatten()
    for i in range(n):
        f = feat[0, i].numpy()
        axes[i].imshow(f, cmap="viridis")
        axes[i].set_title(f"通道 {i}", fontsize=9)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("第一层卷积输出特征图")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130)
    print(f"已保存: {save_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 60)
    print("任务1：CIFAR-10 数据加载与可视化")
    train_loader, test_loader = load_cifar10(
        batch_size=128, root=os.path.join(ROOT, "data"), limit=args.limit)
    show_samples(train_loader, CIFAR10_CLASSES, n=12,
                 save_path=os.path.join(OUT, "exp2_samples.png"))

    print("=" * 60)
    print("任务2~3：训练基础 CNN（depth=2, width=32, kernel=3×3）")
    model = SimpleCNN(depth=2, width=32, kernels=(3, 3))
    hist = train_and_report(model, train_loader, test_loader, args.epochs,
                            "cnn_base")
    plot_confusion(model, test_loader, CIFAR10_CLASSES,
                   save_path=os.path.join(OUT, "exp2_confusion.png"),
                   title="实验二 CNN 混淆矩阵", show=False)
    visualize_kernels(model, os.path.join(OUT, "exp2_kernels.png"))
    visualize_feature_maps(model, test_loader,
                           os.path.join(OUT, "exp2_featuremaps.png"))

    if args.skip_ablation:
        return

    print("=" * 60)
    print("任务4：架构对比（卷积核大小 / 网络深度）")
    e = max(1, args.epochs // 2)
    rows, curves = [], {}
    for tag, kw in [("k3_depth2", dict(depth=2, width=32, kernels=(3, 3))),
                    ("k5_depth2", dict(depth=2, width=32, kernels=(5, 5))),
                    ("k3_depth3", dict(depth=3, width=32, kernels=(3, 3, 3))),
                    ("k3_depth2_w64", dict(depth=2, width=64, kernels=(3, 3)))]:
        m = SimpleCNN(**kw)
        h = train_and_report(m, train_loader, test_loader, e, tag)
        curves[tag] = h["test_acc"]
        rows.append([tag, f"{count_parameters(m):,}", f"{h['best_acc']:.4f}",
                     f"{h['seconds']:.1f}"])

    save_comparison_table(rows, ["结构", "参数量", "最佳测试准确率", "用时(s)"],
                          os.path.join(OUT, "exp2_table_arch.png"),
                          "实验二 CNN 架构对比")

    # 绘制不同结构的测试准确率对比曲线
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag, acc in curves.items():
        ax.plot(range(1, len(acc) + 1), acc, "o-", label=tag)
    ax.set_xlabel("Epoch"); ax.set_ylabel("测试准确率")
    ax.set_title("实验二 不同 CNN 结构测试准确率对比")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp2_arch_curves.png"), dpi=150)
    print(f"已保存: {os.path.join(OUT, 'exp2_arch_curves.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
