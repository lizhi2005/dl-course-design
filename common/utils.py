# -*- coding: utf-8 -*-
"""公共工具模块：随机种子、训练/评估循环、指标统计、绘图。

所有实验（exp1~exp5）与综合应用（app1/app2）共用本模块，
以保证实验设置统一、结果可复现。
"""
import os
import random
import time
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

# Windows 下 matplotlib 中文显示（黑体）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed=SEED):
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(logits, y):
    return (logits.argmax(1) == y).sum().item()


def train_one_epoch(model, loader, loss_fn, optimizer, device=DEVICE,
                    grad_clip=None):
    """训练一个 epoch，返回 (平均损失, 准确率)。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn=None, device=DEVICE):
    """评估，返回 (平均损失, 准确率, 全部预测, 全部标签)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_pred, all_true = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if loss_fn is not None:
            total_loss += loss_fn(logits, y).item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_pred.append(pred.cpu())
        all_true.append(y.cpu())
    loss = total_loss / total if loss_fn is not None else 0.0
    return loss, correct / total, torch.cat(all_pred), torch.cat(all_true)


def fit(model, train_loader, test_loader, epochs=10, lr=1e-3, weight_decay=0.0,
        optimizer_name="adam", scheduler_step=None, grad_clip=None,
        device=DEVICE, save_dir=None, tag="exp", verbose=True):
    """完整训练流程，返回 history 字典（含训练/测试损失与准确率曲线）。"""
    loss_fn = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9,
                                    weight_decay=weight_decay)
    else:
        raise ValueError(f"未知优化器: {optimizer_name}")

    scheduler = None
    if scheduler_step:  # 如 (step_size, gamma)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step[0], gamma=scheduler_step[1])

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc, start = 0.0, time.time()
    for ep in range(1, epochs + 1):
        tr_l, tr_a = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                     device, grad_clip)
        te_l, te_a, _, _ = evaluate(model, test_loader, loss_fn, device)
        history["train_loss"].append(tr_l)
        history["train_acc"].append(tr_a)
        history["test_loss"].append(te_l)
        history["test_acc"].append(te_a)
        best_acc = max(best_acc, te_a)
        if scheduler:
            scheduler.step()
        if verbose:
            print(f"[{tag}] epoch {ep:>2d}/{epochs}  "
                  f"train_loss={tr_l:.4f} train_acc={tr_a:.4f}  "
                  f"test_loss={te_l:.4f} test_acc={te_a:.4f}  "
                  f"({time.time()-start:.1f}s)")
    history["best_acc"] = best_acc
    history["seconds"] = time.time() - start
    history["params"] = count_parameters(model)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f"{tag}.pth"))
        with open(os.path.join(save_dir, f"{tag}_history.json"), "w",
                  encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    return history


def plot_curves(history, save_path=None, title="训练曲线", show=True):
    """绘制损失与准确率曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], "o-", label="训练损失")
    axes[0].plot(history["test_loss"], "s-", label="测试损失")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("损失")
    axes[0].set_title("损失曲线"); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(history["train_acc"], "o-", label="训练准确率")
    axes[1].plot(history["test_acc"], "s-", label="测试准确率")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("准确率")
    axes[1].set_title("准确率曲线"); axes[1].legend(); axes[1].grid(alpha=.3)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"已保存: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion(model, loader, class_names, save_path=None,
                   title="混淆矩阵", normalize=True, show=True):
    """绘制混淆矩阵（不依赖 sklearn 亦可运行，这里用 numpy 手算）。"""
    _, acc, preds, trues = evaluate(model, loader, device=DEVICE)
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for p, t in zip(preds.numpy(), trues.numpy()):
        cm[t, p] += 1
    if normalize:
        with np.errstate(all="ignore"):
            disp = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    else:
        disp = cm.astype(float)

    fig, ax = plt.subplots(figsize=(1.0 * n + 3, 0.9 * n + 2.5))
    im = ax.imshow(disp, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45,
                                                ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title(f"{title}（准确率={acc:.4f}）")
    thresh = disp.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if disp[i, j] > thresh else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"已保存: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return cm


def save_comparison_table(rows, columns, save_path, title="对比结果"):
    """把多组实验结果渲染成表格图片（写报告时可直接插图）。"""
    fig, ax = plt.subplots(figsize=(1.5 * len(columns) + 2,
                                    0.45 * len(rows) + 1.2))
    ax.axis("off")
    ax.set_title(title)
    table = ax.table(cellText=rows, colLabels=columns, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close(fig)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
