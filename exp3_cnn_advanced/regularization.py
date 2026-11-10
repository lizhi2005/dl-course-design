# -*- coding: utf-8 -*-
"""实验三 任务1~2：BatchNorm 与 Dropout 的原理实现与对照实验；数据增强对比。

对照实验：无正则化 / 仅 BatchNorm / 仅 Dropout / BatchNorm+Dropout
数据增强：随机裁剪 + 水平翻转 + 色彩抖动，对比增强前后测试准确率

运行：
    python exp3_cnn_advanced/regularization.py --epochs 8
"""
import os
import sys
import argparse

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data_utils import load_cifar10  # noqa: E402
from common.utils import (set_seed, fit, plot_curves,  # noqa: E402
                          save_comparison_table, count_parameters)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp3_cnn_advanced", "results")


# ------------------------------------------------------------ 手动 BatchNorm
class BatchNormManual:
    """手动实现 Batch Normalization 的前向过程（训练模式）。

    步骤：
        mu = mean(x), sigma^2 = var(x)          （按通道在 N×H×W 上统计）
        x_hat = (x - mu) / sqrt(sigma^2 + eps)
        y = gamma * x_hat + beta                 （可学习缩放与平移）
    推理时使用训练集上累积的移动平均均值/方差。
    """

    @staticmethod
    def forward(x, gamma, beta, eps=1e-5, running_mean=None, running_var=None,
                momentum=0.1, training=True):
        if training:
            mu = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            if running_mean is not None:
                running_mean.copy_(momentum * mu.squeeze()
                                   + (1 - momentum) * running_mean)
                running_var.copy_(momentum * var.squeeze()
                                  + (1 - momentum) * running_var)
        else:
            mu = running_mean.view(1, -1, 1, 1)
            var = running_var.view(1, -1, 1, 1)
        x_hat = (x - mu) / torch.sqrt(var + eps)
        return gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)


def check_batchnorm():
    """验证手写 BatchNorm 与 nn.BatchNorm2d 输出一致。"""
    set_seed(42)
    x = torch.randn(8, 16, 8, 8)
    gamma, beta = torch.randn(16), torch.randn(16)
    rm, rv = torch.zeros(16), torch.ones(16)
    mine = BatchNormManual.forward(x, gamma, beta, running_mean=rm,
                                   running_var=rv, training=True)
    bn = nn.BatchNorm2d(16)
    with torch.no_grad():
        bn.weight.copy_(gamma)
        bn.bias.copy_(beta)
    bn.train()
    ref = bn(x)
    err = (mine - ref).abs().max().item()
    print(f"手写 BatchNorm 与 nn.BatchNorm2d 最大误差: {err:.2e}")
    print(f"  归一化后 通道均值≈{mine.mean(dim=(0,2,3))[0]:.4f} "
          f"（应接近 beta[0]={beta[0]:.4f}）")


class ConvBlock(nn.Module):
    """卷积块：Conv -> [BN] -> ReLU -> Pool -> [Dropout]"""

    def __init__(self, in_c, out_c, use_bn=False, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers += [nn.ReLU(), nn.MaxPool2d(2)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class RegCNN(nn.Module):
    """可开关正则化的 CNN，用于对照实验。"""

    def __init__(self, num_classes=10, use_bn=False, dropout=0.0, width=32):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, width, use_bn, dropout),
            ConvBlock(width, width * 2, use_bn, dropout),
            ConvBlock(width * 2, width * 4, use_bn, dropout))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="限制样本数加速调试，如 20000")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 60)
    print("任务1：手写 BatchNorm 正确性验证")
    check_batchnorm()

    print("=" * 60)
    print("任务1：正则化对照实验（CIFAR-10）")
    train_plain, test_plain = load_cifar10(
        batch_size=128, root=os.path.join(ROOT, "data"), limit=args.limit)
    train_aug, _ = load_cifar10(batch_size=128, root=os.path.join(ROOT, "data"),
                                limit=args.limit, augment=True)

    configs = [("无正则化", dict(use_bn=False, dropout=0.0)),
               ("仅BatchNorm", dict(use_bn=True, dropout=0.0)),
               ("仅Dropout", dict(use_bn=False, dropout=0.3)),
               ("BN+Dropout", dict(use_bn=True, dropout=0.3))]
    rows, curves = [], {}
    for name, kw in configs:
        set_seed(42)
        model = RegCNN(**kw)
        h = fit(model, train_plain, test_plain, epochs=args.epochs, lr=1e-3,
                optimizer_name="adam", tag=f"reg_{name}", verbose=False)
        gap = h["train_acc"][-1] - h["test_acc"][-1]
        print(f"  {name:<12} 参数量={count_parameters(model):>8,}  "
              f"train_acc={h['train_acc'][-1]:.4f}  "
              f"test_acc={h['best_acc']:.4f}  过拟合gap={gap:.4f}")
        curves[name] = h["test_acc"]
        rows.append([name, f"{count_parameters(model):,}",
                     f"{h['train_acc'][-1]:.4f}", f"{h['best_acc']:.4f}",
                     f"{gap:.4f}", f"{h['seconds']:.0f}"])

    save_comparison_table(rows,
                          ["正则化方式", "参数量", "训练准确率", "测试准确率",
                           "泛化差距", "用时(s)"],
                          os.path.join(OUT, "exp3_table_reg.png"),
                          "实验三 正则化对照实验")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, acc in curves.items():
        ax.plot(range(1, len(acc) + 1), acc, "o-", label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("测试准确率")
    ax.set_title("实验三 不同正则化方式测试准确率")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp3_reg_curves.png"), dpi=150)
    plt.close(fig)
    print(f"已保存: {OUT}/exp3_reg_curves.png")

    print("=" * 60)
    print("任务2：数据增强对比（同一网络：BN+Dropout）")
    rows_aug, hist_pair = [], {}
    for name, loader in [("无增强", train_plain), ("有增强", train_aug)]:
        set_seed(42)
        model = RegCNN(use_bn=True, dropout=0.3)
        h = fit(model, loader, test_plain, epochs=args.epochs, lr=1e-3,
                optimizer_name="adam", tag=f"aug_{name}", verbose=False)
        gap = h["train_acc"][-1] - h["test_acc"][-1]
        print(f"  {name:<8} train_acc={h['train_acc'][-1]:.4f}  "
              f"test_acc={h['best_acc']:.4f}  泛化差距={gap:.4f}")
        rows_aug.append([name, f"{h['train_acc'][-1]:.4f}",
                         f"{h['best_acc']:.4f}", f"{gap:.4f}"])
        hist_pair[name] = h

    save_comparison_table(rows_aug,
                          ["数据增强", "训练准确率", "测试准确率", "泛化差距"],
                          os.path.join(OUT, "exp3_table_aug.png"),
                          "实验三 数据增强对比")
    plot_curves(hist_pair["有增强"],
                save_path=os.path.join(OUT, "exp3_aug_curves.png"),
                title="实验三 使用数据增强后的训练曲线", show=False)


if __name__ == "__main__":
    main()
