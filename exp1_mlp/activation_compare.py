# -*- coding: utf-8 -*-
"""实验一 补充：激活函数对比（Sigmoid / Tanh / ReLU）。

指导书实验一"超参数对比"部分除了学习率和批大小，还要求对比激活函数。
这里固定其他条件不变，只换激活函数各跑一遍。

理论预期：Sigmoid 的导数最大只有 0.25，两层隐藏层连乘之后梯度衰减很快，
训练初期收敛会明显慢于 ReLU；Tanh 好一些但同样存在饱和问题。

运行：
    python exp1_mlp/activation_compare.py --epochs 5
"""
import os
import sys
import argparse

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data_utils import load_fashion_mnist  # noqa: E402
from common.utils import set_seed, fit, save_comparison_table, count_parameters  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp1_mlp", "results")


def build_mlp(act_name, hidden1=256, hidden2=128, num_classes=10):
    """按名字取激活函数层。"""
    act = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}[act_name]
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, hidden1), act(),
        nn.Linear(hidden1, hidden2), act(),
        nn.Linear(hidden2, num_classes),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    train_loader, test_loader = load_fashion_mnist(
        batch_size=128, root=os.path.join(ROOT, "data"), limit=args.limit)

    print("=" * 60)
    print("实验一 补充：激活函数对比（hidden=(256,128), lr=1e-3, batch=128）")

    rows, curves = [], {}
    for name in ["sigmoid", "tanh", "relu"]:
        set_seed(42)
        model = build_mlp(name)
        hist = fit(model, train_loader, test_loader, epochs=args.epochs,
                   lr=1e-3, optimizer_name="adam", tag=f"mlp_{name}",
                   verbose=False)
        curves[name] = hist
        rows.append([name, f"{count_parameters(model):,}",
                     f"{hist['best_acc']:.4f}",
                     f"{hist['test_acc'][0]:.4f}",
                     f"{hist['seconds']:.1f}"])
        print(f"   {name:<8} 最佳测试准确率={hist['best_acc']:.4f}  "
              f"第1轮准确率={hist['test_acc'][0]:.4f}  "
              f"用时={hist['seconds']:.1f}s")

    save_comparison_table(
        rows,
        ["激活函数", "参数量", "最佳测试准确率", "第1轮测试准确率", "用时(s)"],
        os.path.join(OUT, "exp1_table_act.png"),
        "实验一 激活函数对比")

    # 把三条曲线画在一起，便于看收敛速度差异
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in curves.items():
        axes[0].plot(h["train_loss"], "o-", label=name, markersize=4)
        axes[1].plot(h["test_acc"], "s-", label=name, markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("训练损失")
    axes[0].set_title("不同激活函数的训练损失")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("测试准确率")
    axes[1].set_title("不同激活函数的测试准确率")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp1_act_curves.png"), dpi=150)
    plt.close(fig)
    print(f"结果已保存到 {OUT}")


if __name__ == "__main__":
    main()
