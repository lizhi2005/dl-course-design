# -*- coding: utf-8 -*-
"""实验四 任务1~2：手动实现 RNN 单元与 LSTM 门控，并演示梯度消失问题。

内容：
    1) 手写 RNN 单元前向：h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)
    2) 手写 LSTM 四个门：遗忘门/输入门/候选状态/输出门
    3) 与 nn.RNN / nn.LSTM 输出对比验证
    4) 梯度消失演示：序列越长，早期时间步的梯度越小

运行：
    python exp4_rnn/rnn_manual.py
"""
import os
import sys
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import set_seed  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp4_rnn", "results")


def rnn_cell_step(x_t, h_prev, W_xh, W_hh, b_h):
    """单个 RNN 单元：h_t = tanh(x_t @ W_xh + h_prev @ W_hh + b_h)"""
    return torch.tanh(x_t @ W_xh + h_prev @ W_hh + b_h)


def rnn_forward_manual(x, h0, W_xh, W_hh, b_h):
    """按时间步展开 RNN。x: (N, T, d_in)，返回所有隐藏状态 (N, T, d_h)。"""
    N, T, _ = x.shape
    d_h = W_hh.shape[0]
    h = h0
    outputs = torch.zeros(N, T, d_h)
    for t in range(T):
        h = rnn_cell_step(x[:, t, :], h, W_xh, W_hh, b_h)
        outputs[:, t, :] = h
    return outputs, h


def lstm_cell_step(x_t, h_prev, c_prev, W_x, W_h, b):
    """LSTM 单元：四个门 + 细胞状态更新。

        i = sigmoid(W_xi x_t + W_hi h_{t-1} + b_i)   输入门
        f = sigmoid(W_xf x_t + W_hf h_{t-1} + b_f)   遗忘门
        g = tanh(   W_xg x_t + W_hg h_{t-1} + b_g)   候选状态
        o = sigmoid(W_xo x_t + W_ho h_{t-1} + b_o)   输出门
        c_t = f * c_{t-1} + i * g
        h_t = o * tanh(c_t)
    """
    gates = x_t @ W_x + h_prev @ W_h + b      # (N, 4*d_h)
    d_h = h_prev.shape[1]
    i, f, g, o = gates[:, :d_h], gates[:, d_h:2 * d_h], \
        gates[:, 2 * d_h:3 * d_h], gates[:, 3 * d_h:]
    i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
    c_t = f * c_prev + i * g
    h_t = o * torch.tanh(c_t)
    return h_t, c_t


def lstm_forward_manual(x, W_x, W_h, b):
    """按时间步展开 LSTM。"""
    N, T, _ = x.shape
    d_h = W_h.shape[0]          # W_h: (d_h, 4*d_h)
    h = torch.zeros(N, d_h)
    c = torch.zeros(N, d_h)
    hs = torch.zeros(N, T, d_h)
    for t in range(T):
        h, c = lstm_cell_step(x[:, t, :], h, c, W_x, W_h, b)
        hs[:, t, :] = h
    return hs, (h, c)


def verify_with_torch():
    """与 PyTorch 内置 RNN/LSTM 对比，验证手写实现正确。"""
    set_seed(42)
    N, T, d_in, d_h = 2, 5, 3, 4

    # --- RNN ---
    rnn = nn.RNN(d_in, d_h, batch_first=True)
    x = torch.randn(N, T, d_in)
    ref_out, _ = rnn(x)
    W_xh = rnn.weight_ih_l0.t()
    W_hh = rnn.weight_hh_l0.t()
    b_h = rnn.bias_hh_l0 + rnn.bias_ih_l0
    mine_out, _ = rnn_forward_manual(x, torch.zeros(N, d_h), W_xh, W_hh, b_h)
    print(f"RNN  手写 vs nn.RNN 最大误差: {(mine_out - ref_out).abs().max():.2e}")

    # --- LSTM ---
    lstm = nn.LSTM(d_in, d_h, batch_first=True)
    ref_out, (ref_h, ref_c) = lstm(x)
    W_x = lstm.weight_ih_l0.t()     # (d_in, 4d_h)
    W_h = lstm.weight_hh_l0.t()     # (d_h , 4d_h)
    b = lstm.bias_hh_l0 + lstm.bias_ih_l0
    mine_out, (mine_h, mine_c) = lstm_forward_manual(x, W_x, W_h, b)
    print(f"LSTM 手写 vs nn.LSTM 最大误差: {(mine_out - ref_out).abs().max():.2e}")
    print(f"      末状态 h 误差: {(mine_h - ref_h.squeeze(0)).abs().max():.2e}")
    return True


def gradient_vanishing_demo(seq_lens=(5, 10, 20, 40, 80), save=True):
    """演示梯度消失：计算损失对第一个时间步输入的梯度范数。"""
    set_seed(42)
    d_in, d_h = 8, 16
    grads_rnn, grads_lstm = [], []
    for T in seq_lens:
        # RNN
        rnn = nn.RNN(d_in, d_h, batch_first=True)
        x = torch.randn(1, T, d_in, requires_grad=True)
        out, _ = rnn(x)
        out[:, -1, :].sum().backward()
        grads_rnn.append(x.grad[:, 0, :].norm().item())

        # LSTM
        lstm = nn.LSTM(d_in, d_h, batch_first=True)
        x2 = torch.randn(1, T, d_in, requires_grad=True)
        o2, _ = lstm(x2)
        o2[:, -1, :].sum().backward()
        grads_lstm.append(x2.grad[:, 0, :].norm().item())
        print(f"  序列长度 T={T:<3}  普通RNN 首步梯度范数={grads_rnn[-1]:.3e}  "
              f"LSTM={grads_lstm[-1]:.3e}")

    if save:
        os.makedirs(OUT, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.semilogy(seq_lens, grads_rnn, "o-", label="普通 RNN")
        ax.semilogy(seq_lens, grads_lstm, "s-", label="LSTM")
        ax.set_xlabel("序列长度 T"); ax.set_ylabel("首时间步梯度范数（对数轴）")
        ax.set_title("实验四 梯度消失现象对比")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "exp4_grad_vanishing.png"), dpi=150)
        plt.close(fig)
        print(f"已保存: {OUT}/exp4_grad_vanishing.png")
    return grads_rnn, grads_lstm


def gradient_clip_demo():
    """梯度裁剪前后对比：构造一个大梯度，观察裁剪效果。"""
    set_seed(42)
    model = nn.Linear(10, 1)
    x = torch.randn(4, 10) * 100      # 放大输入造成大梯度
    y = torch.randn(4, 1)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    before = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    after = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
    print(f"梯度裁剪前总范数={before:.3f}  裁剪(max_norm=1.0)后={after:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("1) 手写 RNN / LSTM 与 PyTorch 实现对比验证")
    verify_with_torch()
    print("=" * 60)
    print("2) 梯度消失现象演示（损失对第一个时间步输入的梯度范数）")
    gradient_vanishing_demo()
    print("=" * 60)
    print("3) 梯度裁剪演示")
    gradient_clip_demo()
