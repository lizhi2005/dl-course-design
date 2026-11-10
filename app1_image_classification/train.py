# -*- coding: utf-8 -*-
"""综合应用一：智能图像分类系统 —— 迁移学习训练与评估。

模型：MobileNetV3-Small（CPU 友好）与 ResNet18（对比用），均加载 ImageNet
      预训练权重，替换分类头后在自定义数据集上微调。

产出：训练曲线、混淆矩阵、最优权重 app1_best.pth、类别名 classes.json

运行：
    python app1_image_classification/train.py --epochs 8 --model mobilenet_v3_small
    python app1_image_classification/train.py --epochs 1 --limit-batch 20  # 冒烟
"""
import os
import sys
import json
import argparse
import time

import torch
import torch.nn as nn
from torchvision import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.utils import (set_seed, fit, plot_curves, plot_confusion,  # noqa: E402
                          save_comparison_table, count_parameters, evaluate)
from app1_image_classification.dataset import get_loaders  # noqa: E402

OUT = os.path.join(ROOT, "app1_image_classification", "results")
CKPT = os.path.join(ROOT, "app1_image_classification", "app1_best.pth")


def build_model(name="mobilenet_v3_small", num_classes=4, pretrained=True):
    """构建迁移学习模型，替换最后的全连接层。"""
    try:
        if name == "mobilenet_v3_small":
            w = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            m = models.mobilenet_v3_small(weights=w)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features,
                                         num_classes)
        elif name == "resnet18":
            w = models.ResNet18_Weights.DEFAULT if pretrained else None
            m = models.resnet18(weights=w)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        elif name == "shufflenet_v2_x1_0":
            w = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
            m = models.shufflenet_v2_x1_0(weights=w)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        else:
            raise ValueError(f"未知模型: {name}")
        print(f"   ✓ {name} 预训练权重加载成功")
    except Exception as e:
        print(f"   ! 预训练权重不可用（{type(e).__name__}），改用随机初始化")
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


class LimitLoader:
    """仅取前 n 个 batch，用于冒烟测试。"""

    def __init__(self, loader, n):
        self.loader, self.n = loader, n

    def __iter__(self):
        for i, b in enumerate(self.loader):
            if i >= self.n:
                break
            yield b

    def __len__(self):
        return min(self.n, len(self.loader))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet_v3_small",
                    choices=["mobilenet_v3_small", "resnet18",
                             "shufflenet_v2_x1_0"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit-batch", type=int, default=None,
                    help="每个 epoch 只用前 N 个 batch（冒烟测试）")
    ap.add_argument("--compare", action="store_true",
                    help="同时训练多个模型做对比")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    set_seed(42)

    train_loader, val_loader, test_loader, classes = get_loaders(
        batch_size=args.batch)
    print(f"数据集类别：{classes}")
    if args.limit_batch:
        train_loader = LimitLoader(train_loader, args.limit_batch)
        val_loader = LimitLoader(val_loader, max(2, args.limit_batch // 2))
        test_loader = LimitLoader(test_loader, max(2, args.limit_batch // 2))

    # 保存类别名，供推理与界面使用
    with open(os.path.join(ROOT, "app1_image_classification", "classes.json"),
              "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False)

    if args.compare:
        rows = []
        for name in ["mobilenet_v3_small", "resnet18"]:
            print("=" * 60)
            print(f"训练模型：{name}")
            m = build_model(name, len(classes))
            h = fit(m, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                    optimizer_name="adam", tag=f"app1_{name}", verbose=False)
            _, te_acc, _, _ = evaluate(m, test_loader)
            print(f"   参数量={count_parameters(m):,}  "
                  f"最佳验证准确率={h['best_acc']:.4f}  "
                  f"测试集准确率={te_acc:.4f}  用时={h['seconds']:.0f}s")
            rows.append([name, f"{count_parameters(m):,}",
                         f"{h['best_acc']:.4f}", f"{te_acc:.4f}",
                         f"{h['seconds']:.0f}"])
        save_comparison_table(rows,
                              ["模型", "参数量", "验证准确率", "测试准确率", "用时(s)"],
                              os.path.join(OUT, "app1_table_models.png"),
                              "综合应用一 模型对比")
        return

    print("=" * 60)
    print(f"训练模型：{args.model}（迁移学习微调）")
    model = build_model(args.model, len(classes))
    print(f"   可训练参数量：{count_parameters(model):,}")
    t0 = time.time()
    history = fit(model, train_loader, val_loader, epochs=args.epochs,
                  lr=args.lr, optimizer_name="adam",
                  scheduler_step=(max(2, args.epochs // 3), 0.5),
                  tag="app1", verbose=True)
    print(f"   最佳验证准确率={history['best_acc']:.4f}  "
          f"总用时={time.time()-t0:.0f}s")
    _, te_acc, _, _ = evaluate(model, test_loader)
    print(f"   测试集准确率={te_acc:.4f}")

    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "model": args.model}, CKPT)
    print(f"模型已保存：{CKPT}")

    plot_curves(history, save_path=os.path.join(OUT, "app1_curves.png"),
                title="综合应用一 训练曲线", show=False)
    # 验证集看泛化差距，测试集出最终成绩（报告里两个都要报）
    plot_confusion(model, val_loader, classes,
                   save_path=os.path.join(OUT, "app1_confusion.png"),
                   title="综合应用一 混淆矩阵（验证集）", show=False)
    plot_confusion(model, test_loader, classes,
                   save_path=os.path.join(OUT, "app1_confusion_test.png"),
                   title="综合应用一 混淆矩阵（测试集）", show=False)


if __name__ == "__main__":
    main()
