# -*- coding: utf-8 -*-
"""实验一 任务2：从零实现两层全连接网络（手写前向 / 反向 / SGD）。

仅使用 torch 张量运算，不依赖 autograd，用于理解反向传播的梯度公式。
数据集：Fashion-MNIST（28×28 灰度图，10 类）

运行：
    python exp1_mlp/mlp_scratch.py --epochs 5
"""
import os
import sys
import argparse
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data_utils import load_fashion_mnist, FASHION_CLASSES  # noqa: E402
from common.utils import set_seed, plot_curves  # noqa: E402


# ---------------------------------------------------------------- 从零实现
class MLPScratch:
    """两层 MLP：X(784) -> 256(ReLU) -> 10，手动实现前向、反向与 SGD。"""

    def __init__(self, in_dim=784, hidden=256, num_classes=10, seed=42):
        g = torch.Generator().manual_seed(seed)
        # He 初始化：ReLU 网络用 sqrt(2/fan_in) 更稳定
        self.W1 = torch.randn(in_dim, hidden, generator=g) * (2.0 / in_dim) ** 0.5
        self.b1 = torch.zeros(hidden)
        self.W2 = torch.randn(hidden, num_classes, generator=g) * (2.0 / hidden) ** 0.5
        self.b2 = torch.zeros(num_classes)

    def forward(self, X):
        """X: (N, 784)。返回 logits 与中间变量（供反向传播使用）。"""
        self.X = X
        self.z1 = X @ self.W1 + self.b1          # (N, hidden)
        self.h1 = torch.relu(self.z1)            # ReLU 激活
        self.z2 = self.h1 @ self.W2 + self.b2    # (N, 10)
        return self.z2

    def backward(self, logits, y):
        """计算 Softmax 交叉熵损失与全部参数梯度。

        推导：
            p = softmax(logits)
            L = -log(p[y])
            dL/dz2 = (p - onehot(y)) / N
            dL/dW2 = h1^T @ dL/dz2
            dL/dh1 = dL/dz2 @ W2^T
            dL/dz1 = dL/dh1 * [z1 > 0]        (ReLU 导数为 0/1)
            dL/dW1 = X^T @ dL/dz1
        """
        N = logits.size(0)
        # 数值稳定的 softmax：先减去最大值，避免 exp 溢出
        z = logits - logits.max(dim=1, keepdim=True).values
        exp = torch.exp(z)
        probs = exp / exp.sum(dim=1, keepdim=True)
        loss = -torch.log(probs[torch.arange(N), y] + 1e-8).mean()

        dz2 = probs.clone()
        dz2[torch.arange(N), y] -= 1.0
        dz2 /= N

        self.dW2 = self.h1.t() @ dz2
        self.db2 = dz2.sum(0)
        dh1 = dz2 @ self.W2.t()
        dz1 = dh1 * (self.z1 > 0).float()        # ReLU 导数
        self.dW1 = self.X.t() @ dz1
        self.db1 = dz1.sum(0)
        return loss.item()

    def sgd_step(self, lr):
        with torch.no_grad():
            self.W1 -= lr * self.dW1
            self.b1 -= lr * self.db1
            self.W2 -= lr * self.dW2
            self.b2 -= lr * self.db2

    def params(self):
        return self.W1.numel() + self.b1.numel() + self.W2.numel() + self.b2.numel()


def train_scratch(model, train_loader, test_loader, epochs=5, lr=0.1):
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    for ep in range(1, epochs + 1):
        model_grad_time = time.time()
        tot_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            X = x.flatten(1)
            logits = model.forward(X)
            loss = model.backward(logits, y)
            model.sgd_step(lr)
            tot_loss += loss * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)
        # 测试集评估
        tc, tt = 0, 0
        for x, y in test_loader:
            logits = model.forward(x.flatten(1))
            tc += (logits.argmax(1) == y).sum().item()
            tt += y.size(0)
        history["train_loss"].append(tot_loss / total)
        history["train_acc"].append(correct / total)
        history["test_acc"].append(tc / tt)
        print(f"[从零实现] epoch {ep}/{epochs}  loss={tot_loss/total:.4f}  "
              f"train_acc={correct/total:.4f}  test_acc={tc/tt:.4f}  "
              f"({time.time()-model_grad_time:.1f}s)")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None,
                    help="限制训练样本数，用于快速调试")
    args = ap.parse_args()

    set_seed(42)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_loader, test_loader = load_fashion_mnist(
        batch_size=args.batch, root=os.path.join(root, "data"), limit=args.limit)

    model = MLPScratch(hidden=args.hidden)
    print(f"从零实现 MLP 参数量：{model.params()}")
    history = train_scratch(model, train_loader, test_loader,
                            epochs=args.epochs, lr=args.lr)

    out = os.path.join(root, "exp1_mlp", "results")
    os.makedirs(out, exist_ok=True)
    plot_curves({"train_loss": history["train_loss"],
                 "test_loss": history["train_loss"],
                 "train_acc": history["train_acc"],
                 "test_acc": history["test_acc"]},
                save_path=os.path.join(out, "exp1_scratch_curves.png"),
                title="实验一 从零实现 MLP 训练曲线", show=False)


if __name__ == "__main__":
    main()
