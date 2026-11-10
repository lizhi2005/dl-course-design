# -*- coding: utf-8 -*-
"""实验二 任务1：手动实现 2D 卷积与池化，并与 PyTorch 实现对比验证。

知识点：滑动窗口、逐元素相乘求和、步长(stride)、填充(padding)、
        最大池化 / 平均池化、多通道卷积的输出尺寸计算。

运行：
    python exp2_cnn/conv_manual.py
"""
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import set_seed  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def conv2d_manual(x, kernel, bias=None, stride=1, padding=0):
    """手动实现 2D 互相关（深度学习中的"卷积"）。

    参数：
        x:      (C_in, H, W) 或 (N, C_in, H, W)
        kernel: (C_out, C_in, K_h, K_w)
    返回：
        out:    与输入同维数，形状为 (..., C_out, H_out, W_out)
                H_out = (H + 2P - K) / S + 1
    """
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)
    N, C_in, H, W = x.shape
    C_out, C_in_k, K_h, K_w = kernel.shape
    assert C_in == C_in_k, "输入通道数与卷积核通道数不一致"

    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding))
    _, _, Hp, Wp = x.shape
    H_out = (Hp - K_h) // stride + 1
    W_out = (Wp - K_w) // stride + 1
    assert H_out > 0 and W_out > 0, "卷积核大于输入尺寸，请检查 padding/stride"

    # im2col：把每个滑动窗口展成一行，用矩阵乘法一次性完成卷积
    patches = []
    for i in range(H_out):
        for j in range(W_out):
            win = x[:, :,
                    i * stride:i * stride + K_h,
                    j * stride:j * stride + K_w]        # (N, C_in, K_h, K_w)
            patches.append(win.reshape(N, -1))          # (N, C_in*K_h*K_w)
    patches = torch.stack(patches, dim=1)               # (N, L, C_in*K_h*K_w)
    kmat = kernel.reshape(C_out, -1)                    # (C_out, C_in*K_h*K_w)
    out = patches @ kmat.t()                            # (N, L, C_out)
    out = out.transpose(1, 2).reshape(N, C_out, H_out, W_out)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out.squeeze(0) if single else out


def max_pool2d_manual(x, kernel_size=2, stride=None, return_mask=False):
    """手动实现最大池化。stride 默认为 kernel_size。"""
    stride = stride or kernel_size
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)
    N, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    windows = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    windows = windows.contiguous().view(N, C, H_out, W_out, -1)
    values, idx = windows.max(dim=-1)
    if single:
        values = values.squeeze(0)
    return (values, idx) if return_mask else values


def avg_pool2d_manual(x, kernel_size=2, stride=None):
    """手动实现平均池化。"""
    stride = stride or kernel_size
    single = (x.dim() == 3)
    if single:
        x = x.unsqueeze(0)
    N, C, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    windows = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    out = windows.contiguous().view(N, C, H_out, W_out, -1).mean(dim=-1)
    return out.squeeze(0) if single else out


def demo():
    set_seed(42)
    print("=" * 60)
    print("1) 单通道 5×5 输入，3×3 卷积核，不同 stride / padding 对比")
    x = torch.arange(25, dtype=torch.float32).reshape(1, 5, 5)
    k = torch.ones(1, 1, 3, 3)
    for stride in (1, 2):
        for padding in (0, 1):
            mine = conv2d_manual(x, k, stride=stride, padding=padding)
            ref = F.conv2d(x.unsqueeze(0), k, stride=stride,
                           padding=padding).squeeze(0)
            err = (mine - ref).abs().max().item()
            print(f"   stride={stride} padding={padding} -> 输出尺寸 "
                  f"{tuple(mine.shape)}  与 torch 最大误差={err:.2e}")

    print("=" * 60)
    print("2) 多通道卷积（3 通道输入 -> 4 通道输出）与 torch 对比")
    x3 = torch.randn(2, 3, 16, 16)
    k4 = torch.randn(4, 3, 3, 3)
    b4 = torch.randn(4)
    mine = conv2d_manual(x3, k4, bias=b4, stride=1, padding=1)
    ref = F.conv2d(x3, k4, bias=b4, stride=1, padding=1)
    print(f"   输出尺寸 {tuple(mine.shape)}，与 torch 最大误差 "
          f"={(mine - ref).abs().max().item():.2e}")

    print("=" * 60)
    print("3) 池化对比")
    xp = torch.randn(1, 2, 8, 8)
    mp_mine = max_pool2d_manual(xp, 2)
    mp_ref = F.max_pool2d(xp, 2)
    ap_mine = avg_pool2d_manual(xp, 2)
    ap_ref = F.avg_pool2d(xp, 2)
    print(f"   最大池化 形状{tuple(mp_mine.shape)} 误差"
          f"={(mp_mine - mp_ref).abs().max().item():.2e}")
    print(f"   平均池化 形状{tuple(ap_mine.shape)} 误差"
          f"={(ap_mine - ap_ref).abs().max().item():.2e}")

    print("=" * 60)
    print("4) 感受野示例：两次 3×3 卷积（stride=1）等价于 5×5 感受野")
    xx = torch.randn(1, 1, 7, 7)
    kk = torch.randn(1, 1, 3, 3)
    o1 = conv2d_manual(xx, kk)          # 7 -> 5
    o2 = conv2d_manual(o1, kk)          # 5 -> 3
    print(f"   输入 7×7 -> 第一层 {tuple(o1.shape[1:])} -> "
          f"第二层 {tuple(o2.shape[1:])}；第二层每个神经元覆盖输入 5×5 区域")

    print("=" * 60)
    print("5) 输出尺寸公式验证：H_out = (H + 2P - K)/S + 1")
    for H, K, P, S in [(32, 3, 1, 1), (32, 5, 0, 1), (32, 3, 0, 2), (28, 5, 2, 1)]:
        print(f"   H={H} K={K} P={P} S={S} -> H_out={(H + 2*P - K)//S + 1}")


if __name__ == "__main__":
    demo()
