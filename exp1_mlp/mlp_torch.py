# -*- coding: utf-8 -*-
"""实验一 任务3~5：用 nn.Module 实现 MLP，并完成超参数对比实验。

对比三组超参数：学习率、隐藏单元数、批量大小；结果输出为对比表与曲线图。

运行：
    python exp1_mlp/mlp_torch.py --epochs 10
    python exp1_mlp/mlp_torch.py --epochs 1 --limit 20000   # 快速调试
"""
import os
import sys
import argparse

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data_utils import load_fashion_mnist  # noqa: E402
from common.utils import (set_seed, fit, plot_curves, plot_confusion,  # noqa: E402
                          save_comparison_table, count_parameters)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MLP(nn.Module):
    """三层全连接网络：784 -> hidden1 -> hidden2 -> num_classes"""

    def __init__(self, in_dim=28 * 28, hidden1=256, hidden2=128,
                 num_classes=10, dropout=0.0):
        super().__init__()
        layers = [nn.Flatten(),
                  nn.Linear(in_dim, hidden1), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden1, hidden2), nn.ReLU(),
                   nn.Linear(hidden2, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_one(train_loader, test_loader, lr=1e-3, hidden1=256, hidden2=128,
            batch=128, epochs=5, tag="mlp"):
    set_seed(42)
    model = MLP(hidden1=hidden1, hidden2=hidden2).to("cpu")
    history = fit(model, train_loader, test_loader, epochs=epochs, lr=lr,
                  optimizer_name="adam", tag=tag, verbose=False)
    print(f"  设置 lr={lr:<6} hidden=({hidden1},{hidden2}) batch={batch:<4} -> "
          f"best_test_acc={history['best_acc']:.4f}  "
          f"参数量={count_parameters(model):,}  用时={history['seconds']:.1f}s")
    return history, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-grid", action="store_true", help="跳过超参数对比")
    args = ap.parse_args()

    out = os.path.join(ROOT, "exp1_mlp", "results")
    os.makedirs(out, exist_ok=True)

    # ---------- 任务3/4：基线模型训练 ----------
    print("=" * 60)
    print("任务3~4：nn.Module 版 MLP 基线训练（Fashion-MNIST）")
    train_loader, test_loader = load_fashion_mnist(
        batch_size=128, root=os.path.join(ROOT, "data"), limit=args.limit)
    base_hist, base_model = run_one(train_loader, test_loader, lr=1e-3,
                                    epochs=args.epochs, tag="mlp_base")
    plot_curves(base_hist, save_path=os.path.join(out, "exp1_mlp_curves.png"),
                title="实验一 MLP 训练曲线（nn.Module）", show=False)
    plot_confusion(base_model, test_loader,
                   [str(i) for i in range(10)],
                   save_path=os.path.join(out, "exp1_mlp_confusion.png"),
                   title="实验一 MLP 混淆矩阵", show=False)

    if args.skip_grid:
        return

    # ---------- 任务5：超参数对比 ----------
    print("=" * 60)
    print("任务5：超参数对比实验")
    grid_epochs = max(1, args.epochs // 2)

    groups = []
    # (1) 学习率
    print("(1) 学习率对比：", end="")
    lrs = [0.001, 0.01, 0.1, 1.0]
    print(lrs)
    rows_lr = []
    for lr in lrs:
        h, _ = run_one(train_loader, test_loader, lr=lr, epochs=grid_epochs,
                       tag=f"mlp_lr{lr}")
        rows_lr.append([lr, 256, 128, f"{h['best_acc']:.4f}",
                        f"{h['train_loss'][-1]:.4f}", f"{h['seconds']:.1f}"])
    save_comparison_table(rows_lr,
                          ["学习率", "隐藏层1", "隐藏层2", "最佳测试准确率",
                           "最终训练损失", "用时(s)"],
                          os.path.join(out, "exp1_table_lr.png"),
                          "实验一 学习率对比")

    # (2) 隐藏单元数
    print("(2) 隐藏单元数对比：", end="")
    hiddens = [64, 256, 1024]
    print(hiddens)
    rows_h = []
    for hd in hiddens:
        h, _ = run_one(train_loader, test_loader, lr=1e-3, hidden1=hd,
                       hidden2=max(32, hd // 2), epochs=grid_epochs,
                       tag=f"mlp_h{hd}")
        rows_h.append([1e-3, hd, max(32, hd // 2), f"{h['best_acc']:.4f}",
                       f"{h['train_loss'][-1]:.4f}", f"{h['seconds']:.1f}"])
    save_comparison_table(rows_h,
                          ["学习率", "隐藏层1", "隐藏层2", "最佳测试准确率",
                           "最终训练损失", "用时(s)"],
                          os.path.join(out, "exp1_table_hidden.png"),
                          "实验一 隐藏单元数对比")

    # (3) 批量大小
    print("(3) 批量大小对比：", end="")
    batches = [32, 128, 512]
    print(batches)
    rows_b = []
    for bs in batches:
        tr, te = load_fashion_mnist(batch_size=bs,
                                    root=os.path.join(ROOT, "data"),
                                    limit=args.limit)
        h, _ = run_one(tr, te, lr=1e-3, batch=bs, epochs=grid_epochs,
                       tag=f"mlp_b{bs}")
        rows_b.append([bs, 256, 128, f"{h['best_acc']:.4f}",
                       f"{h['train_loss'][-1]:.4f}", f"{h['seconds']:.1f}"])
    save_comparison_table(rows_b,
                          ["批量大小", "隐藏层1", "隐藏层2", "最佳测试准确率",
                           "最终训练损失", "用时(s)"],
                          os.path.join(out, "exp1_table_batch.png"),
                          "实验一 批量大小对比")

    print("=" * 60)
    print(f"全部结果已保存到: {out}")
    print("请将表格与曲线截图插入报告第四章。")


if __name__ == "__main__":
    main()
