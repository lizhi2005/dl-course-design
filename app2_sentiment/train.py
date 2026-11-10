# -*- coding: utf-8 -*-
"""综合应用二：智能文本情感分析系统 —— BERT 微调训练与评估。

流程：数据准备 -> 文本预处理 -> BERT 微调 -> 评估（准确率/F1/混淆矩阵）
      -> 保存模型（HuggingFace 格式，供界面直接加载）

CPU 提示：默认 --limit 800、--max-length 64、--batch 8，1—2 个 epoch。

运行：
    python app2_sentiment/train.py --epochs 2 --limit 800
    python app2_sentiment/train.py --epochs 1 --limit 100   # 冒烟测试
"""
import os
import sys
import json
import argparse
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.utils import (set_seed, plot_confusion, save_comparison_table,  # noqa: E402
                          plot_curves)
from common.nlp_utils import load_chn_split  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
OUT = os.path.join(ROOT, "app2_sentiment", "results")
SAVE_DIR = os.path.join(ROOT, "app2_sentiment", "model")
BERT_DIR = os.path.join(ROOT, "models", "bert-base-chinese")


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.encodings.items()}, self.labels[i]


def f1_binary(pred, true):
    tp = ((pred == 1) & (true == 1)).sum().item()
    fp = ((pred == 1) & (true == 0)).sum().item()
    fn = ((pred == 0) & (true == 1)).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None,
                    help="训练样本上限；正式实验用全量（约 9600 条）")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-ratio", type=float, default=0.15,
                    help="从训练集里切出的验证集比例")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    set_seed(42)

    # ---------------- 1) 数据与预处理 ----------------
    print("=" * 60)
    print("步骤1：数据准备与文本预处理")
    tr_x, tr_y, te_x, te_y = load_chn_split(limit=args.limit)
    # 指导书要求 train / val / test 三个划分，这里从训练集里再切一块做验证，
    # 测试集始终用官方切分，不参与任何调参。
    n_val = int(len(tr_x) * args.val_ratio)
    va_x, va_y = tr_x[:n_val], tr_y[:n_val]
    tr_x, tr_y = tr_x[n_val:], tr_y[n_val:]
    print(f"   训练集 {len(tr_x)} 条 / 验证集 {len(va_x)} 条 / 测试集 {len(te_x)} 条")
    print(f"   样例：{'正面' if tr_y[0]==1 else '负面'} | {tr_x[0][:40]}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_path = BERT_DIR if os.path.exists(BERT_DIR) else "bert-base-chinese"
    print(f"   加载预训练模型：{model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               num_labels=2)

    enc_tr = tok(list(tr_x), truncation=True, padding="max_length",
                 max_length=args.max_length, return_tensors="pt")
    enc_va = tok(list(va_x), truncation=True, padding="max_length",
                 max_length=args.max_length, return_tensors="pt")
    enc_te = tok(list(te_x), truncation=True, padding="max_length",
                 max_length=args.max_length, return_tensors="pt")
    train_loader = torch.utils.data.DataLoader(
        TextDataset(enc_tr, torch.tensor(tr_y)), batch_size=args.batch,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        TextDataset(enc_va, torch.tensor(va_y)), batch_size=args.batch)
    test_loader = torch.utils.data.DataLoader(
        TextDataset(enc_te, torch.tensor(te_y)), batch_size=args.batch)

    @torch.no_grad()
    def quick_acc(loader):
        model.eval()
        c, n = 0, 0
        for batch, y in loader:
            c += (model(**batch).logits.argmax(-1) == y).sum().item()
            n += len(y)
        model.train()
        return c / max(n, 1)

    # ---------------- 2) 微调训练 ----------------
    print("=" * 60)
    print("步骤2：BERT 微调训练")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    t0 = time.time()
    history = {"train_loss": [], "test_acc": []}
    for ep in range(1, args.epochs + 1):
        total, correct, n = 0.0, 0, 0
        for i, (batch, y) in enumerate(train_loader, 1):
            out = model(**batch, labels=y)
            loss = out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            correct += (out.logits.argmax(-1) == y).sum().item()
            n += len(y)
            if i % 20 == 0:
                print(f"      epoch {ep} batch {i}/{len(train_loader)} "
                      f"loss={total/i:.4f} acc={correct/n:.4f} "
                      f"({time.time()-t0:.0f}s)")
        acc = correct / n
        va_acc = quick_acc(val_loader)
        history["train_loss"].append(total / len(train_loader))
        history["test_acc"].append(va_acc)
        print(f"   epoch {ep}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"train_acc={acc:.4f}  val_acc={va_acc:.4f}  "
              f"用时={time.time()-t0:.0f}s")

    # ---------------- 3) 评估 ----------------
    print("=" * 60)
    print("步骤3：测试集评估")
    model.eval()
    preds = []
    with torch.no_grad():
        for batch, _ in test_loader:
            logits = model(**batch).logits
            preds.append(logits.argmax(-1))
    pred = torch.cat(preds)
    true = torch.tensor(te_y)
    acc = (pred == true).float().mean().item()
    prec, rec, f1 = f1_binary(pred, true)
    print(f"   验证集准确率={quick_acc(val_loader):.4f}")
    print(f"   测试集：准确率={acc:.4f}  精确率={prec:.4f}  召回率={rec:.4f}  "
          f"F1={f1:.4f}")

    # 混淆矩阵
    cm = torch.zeros(2, 2, dtype=torch.int64)
    for p, t in zip(pred, true):
        cm[t.item(), p.item()] += 1
    print(f"   混淆矩阵（行=真实，列=预测）：\n{cm.tolist()}")

    # ---------------- 4) 保存 ----------------
    print("=" * 60)
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    meta = {"acc": acc, "f1": f1, "epochs": args.epochs,
            "train_size": len(tr_x), "test_size": len(te_x)}
    with open(os.path.join(SAVE_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"模型已保存到：{SAVE_DIR}")

    save_comparison_table([["BERT 微调", f"{acc:.4f}", f"{prec:.4f}",
                            f"{rec:.4f}", f"{f1:.4f}"]],
                          ["方法", "准确率", "精确率", "召回率", "F1"],
                          os.path.join(OUT, "app2_result.png"),
                          "综合应用二 情感分析结果")
    plot_curves({"train_loss": history["train_loss"],
                 "test_loss": history["train_loss"],
                 "train_acc": history["test_acc"],
                 "test_acc": history["test_acc"]},
                save_path=os.path.join(OUT, "app2_curves.png"),
                title="综合应用二 BERT 微调曲线", show=False)

    # 示例预测
    print("=" * 60)
    print("示例预测：")
    for s in ["房间干净整洁，服务热情，非常满意",
              "设施陈旧，隔音很差，完全不推荐"]:
        enc = tok(s, truncation=True, padding="max_length",
                  max_length=args.max_length, return_tensors="pt")
        with torch.no_grad():
            p = F.softmax(model(**enc).logits, dim=1)[0]
        print(f"   '{s}' -> {'正面' if p.argmax().item()==1 else '负面'} "
              f"(置信度 {p.max().item()*100:.2f}%)")


if __name__ == "__main__":
    main()
