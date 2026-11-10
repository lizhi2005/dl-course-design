# -*- coding: utf-8 -*-
"""实验四 任务3~4：文本预处理 + LSTM 情感分类 + RNN/LSTM/GRU 对比。

运行：
    python exp4_rnn/imdb_lstm.py --epochs 5
    python exp4_rnn/imdb_lstm.py --epochs 1 --limit 2000 --max-len 60  # 快速调试
"""
import os
import sys
import argparse
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.nlp_utils import load_imdb, tokenize_en, encode  # noqa: E402
from common.utils import (set_seed, train_one_epoch, evaluate, plot_curves,  # noqa: E402
                          save_comparison_table, count_parameters)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp4_rnn", "results")


class RNNClassifier(nn.Module):
    """通用循环网络分类器，rnn_type 可选 rnn / lstm / gru。"""

    def __init__(self, vocab_size, embed_dim=128, hidden=128, num_layers=2,
                 num_classes=2, pad_idx=0, rnn_type="lstm", dropout=0.3,
                 pooling="last"):
        super().__init__()
        self.rnn_type = rnn_type
        self.pooling = pooling
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(embed_dim, hidden, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        e = self.dropout(self.embed(x))            # (N, T, E)
        out, h = self.rnn(e)                       # out: (N, T, H)
        if self.pooling == "last":
            feat = out[:, -1, :]                   # 末时间步
        else:                                      # mean：所有时间步平均
            mask = (x != 0).float().unsqueeze(-1)
            feat = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.dropout(feat))


def train_model(model, train_loader, test_loader, epochs=5, lr=1e-3,
                grad_clip=1.0, tag="lstm"):
    set_seed(42)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best, start = 0.0, time.time()
    for ep in range(1, epochs + 1):
        tr_l, tr_a = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                     grad_clip=grad_clip)
        te_l, te_a, _, _ = evaluate(model, test_loader, loss_fn)
        history["train_loss"].append(tr_l); history["train_acc"].append(tr_a)
        history["test_loss"].append(te_l); history["test_acc"].append(te_a)
        best = max(best, te_a)
        print(f"  [{tag}] epoch {ep}/{epochs}  loss={tr_l:.4f} "
              f"train_acc={tr_a:.4f}  test_acc={te_a:.4f}  "
              f"({time.time()-start:.0f}s)")
    history["best_acc"] = best
    history["seconds"] = time.time() - start
    history["params"] = count_parameters(model)
    return history


def f1_score(model, loader):
    """手工计算二分类 F1（不依赖 sklearn）。"""
    _, acc, preds, trues = evaluate(model, loader)
    p, t = preds.numpy(), trues.numpy()
    tp = ((p == 1) & (t == 1)).sum()
    fp = ((p == 1) & (t == 0)).sum()
    fn = ((p == 0) & (t == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return acc, precision, recall, f1


def predict_sentiment(model, text, vocab, max_len=100):
    """对新文本做情感预测（供演示与交互界面使用）。"""
    model.eval()
    ids = encode([tokenize_en(text)], vocab, max_len)
    with torch.no_grad():
        logits = model(ids)
        prob = torch.softmax(logits, dim=1)[0]
        pred = int(prob.argmax())
    return ("正面" if pred == 1 else "负面"), float(prob[pred])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--limit", type=int, default=5000, help="训练样本数上限")
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--skip-compare", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 60)
    print("任务3：文本数据准备（分词 / 词表 / 填充）")
    train_loader, test_loader, vocab = load_imdb(
        max_len=args.max_len, batch_size=args.batch, limit=args.limit)
    print(f"   词表大小: {len(vocab)}   序列长度: {args.max_len}")

    print("=" * 60)
    print("任务3：训练 LSTM 情感分类模型")
    model = RNNClassifier(len(vocab) + 2, rnn_type="lstm")
    print(f"   模型参数量: {count_parameters(model):,}")
    hist = train_model(model, train_loader, test_loader, epochs=args.epochs,
                       tag="lstm")
    plot_curves(hist, save_path=os.path.join(OUT, "exp4_lstm_curves.png"),
                title="实验四 LSTM 情感分类训练曲线", show=False)

    acc, prec, rec, f1 = f1_score(model, test_loader)
    print(f"   测试集：准确率={acc:.4f} 精确率={prec:.4f} "
          f"召回率={rec:.4f} F1={f1:.4f}")

    print("=" * 60)
    print("示例预测：")
    for s in ["this movie is great and touching",
              "boring and awful, waste of time"]:
        label, conf = predict_sentiment(model, s, vocab, args.max_len)
        print(f"   '{s}' -> {label} (置信度 {conf:.3f})")

    if args.skip_compare:
        return

    print("=" * 60)
    print("任务4：RNN / LSTM / GRU 对比实验")
    rows, curves = [], {}
    for rtype in ["rnn", "lstm", "gru"]:
        m = RNNClassifier(len(vocab) + 2, rnn_type=rtype)
        h = train_model(m, train_loader, test_loader, epochs=args.epochs,
                        tag=rtype)
        a, p, r, f = f1_score(m, test_loader)
        print(f"   {rtype.upper():<5} 参数量={count_parameters(m):,} "
              f"acc={a:.4f} F1={f:.4f} 用时={h['seconds']:.0f}s")
        rows.append([rtype.upper(), f"{count_parameters(m):,}", f"{a:.4f}",
                     f"{f:.4f}", f"{h['seconds']:.0f}"])
        curves[rtype.upper()] = h["test_acc"]

    save_comparison_table(rows, ["模型", "参数量", "准确率", "F1", "用时(s)"],
                          os.path.join(OUT, "exp4_table_models.png"),
                          "实验四 RNN / LSTM / GRU 对比")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, v in curves.items():
        ax.plot(range(1, len(v) + 1), v, "o-", label=k)
    ax.set_xlabel("Epoch"); ax.set_ylabel("测试准确率")
    ax.set_title("实验四 三种循环网络测试准确率对比")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp4_compare_curves.png"), dpi=150)
    plt.close(fig)
    print(f"已保存: {OUT}/exp4_compare_curves.png")


if __name__ == "__main__":
    main()
