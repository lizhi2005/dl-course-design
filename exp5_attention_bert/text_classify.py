# -*- coding: utf-8 -*-
"""实验五 补充：LSTM / LSTM+Attention / Transformer 三种文本分类模型。

指导书实验五结尾要求把四种模型放一起对比：
    LSTM  /  LSTM+Attention  /  Transformer  /  BERT
其中 BERT 在 bert_finetune.py 里已经单独跑过，这里补齐前三个，
用同一份中文情感数据、同一套评估口径，结果可以直接并到一张表里。

三个模型共用词表和数据，差别只在编码器：
    1) LSTM              —— 取最后一个有效时间步的隐状态
    2) LSTM + 加性注意力  —— 对全部时间步加权求和，权重可导出看
    3) Transformer 编码器 —— 正弦位置编码 + 自注意力 + 掩码平均池化

运行：
    python exp5_attention_bert/text_classify.py --epochs 5
    python exp5_attention_bert/text_classify.py --epochs 2 --limit 2000   # 冒烟
"""
import os
import sys
import math
import time
import argparse
from collections import Counter

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import set_seed, count_parameters  # noqa: E402
from common.nlp_utils import load_chn_split  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp5_attention_bert", "results")
PAD, UNK = 0, 1


# --------------------------------------------------------------- 数据
def zh_tokenize(texts):
    """结巴分词。jieba 第一次导入比较慢，就放在函数里只 import 一次。"""
    import jieba
    jieba.setLogLevel(20)          # 关掉 "Building prefix dict" 那堆日志
    return [list(jieba.cut(t)) for t in texts]


def build_vocab(token_lists, min_freq=2, max_size=30000):
    cnt = Counter()
    for toks in token_lists:
        cnt.update(toks)
    words = [w for w, c in cnt.most_common(max_size) if c >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(token_lists, vocab, max_len=64):
    ids = []
    for toks in token_lists:
        seq = [vocab.get(w, UNK) for w in toks][:max_len]
        seq += [PAD] * (max_len - len(seq))
        ids.append(seq)
    return torch.tensor(ids, dtype=torch.long)


def make_loader(ids, labels, batch_size=64, shuffle=False):
    ds = torch.utils.data.TensorDataset(ids, torch.tensor(labels, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------- 三个模型
def last_valid_step(out, mask):
    """取每个样本"最后一个非 pad 位置"的隐状态。

    坑：padding 都补在句子后面，直接写 out[:, -1, :] 拿到的是 pad 位置的状态。
    LSTM 在 pad 上还会继续更新，虽然影响有限，但语义上不对，
    所以老老实实按长度索引。
    """
    length = mask.long().sum(1).clamp(min=1) - 1
    return out[torch.arange(out.size(0)), length]


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=64, layers=2, n_cls=2,
                 dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_cls)

    def forward(self, x, return_attn=False):
        mask = x != PAD
        out, _ = self.lstm(self.emb(x))
        feat = self.drop(last_valid_step(out, mask))
        logits = self.fc(feat)
        return (logits, None) if return_attn else logits


class LSTMAttentionClassifier(nn.Module):
    """在 LSTM 输出上加一层加性注意力（Bahdanau 形式）。"""

    def __init__(self, vocab_size, emb=128, hidden=64, layers=2, n_cls=2,
                 dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.score = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_cls)

    def forward(self, x, return_attn=False):
        mask = x != PAD
        out, _ = self.lstm(self.emb(x))
        s = self.score(out).squeeze(-1)              # (N, T)
        # pad 位置要屏蔽掉，否则 softmax 会把权重分给 <pad>
        s = s.masked_fill(~mask, -1e9)
        w = torch.softmax(s, dim=-1)
        feat = self.drop(torch.bmm(w.unsqueeze(1), out).squeeze(1))
        logits = self.fc(feat)
        return (logits, w) if return_attn else logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, layers=2, n_cls=2,
                 max_len=128, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pe = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model, nhead,
                                           dim_feedforward=256,
                                           dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.fc = nn.Linear(d_model, n_cls)

    def forward(self, x, return_attn=False):
        mask = x != PAD
        h = self.pe(self.emb(x))
        h = self.encoder(h, src_key_padding_mask=~mask)
        # 掩码平均池化，忽略 pad
        h = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        logits = self.fc(h)
        return (logits, None) if return_attn else logits


# --------------------------------------------------------------- 训练/评估
def train_model(model, tr_loader, va_loader, epochs=5, lr=1e-3, tag="model",
                grad_clip=1.0):
    set_seed(42)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_acc": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total, cnt = 0.0, 0
        for x, y in tr_loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += loss.item()
            cnt += 1
        acc = evaluate(model, va_loader)[0]
        hist["train_loss"].append(total / cnt)
        hist["val_acc"].append(acc)
        print(f"   [{tag}] epoch {ep}/{epochs}  loss={total/cnt:.4f}  "
              f"val_acc={acc:.4f}  ({time.time()-t0:.0f}s)")
    hist["seconds"] = time.time() - t0
    return hist


@torch.no_grad()
def evaluate(model, loader):
    """返回 (准确率, 精确率, 召回率, F1)。正类取 label=1。"""
    model.eval()
    tp = fp = fn = correct = total = 0
    for x, y in loader:
        pred = model(x).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()
    acc = correct / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return acc, prec, rec, f1


# --------------------------------------------------------------- 可视化
def plot_attention_examples(model, texts, vocab, max_len, save_path,
                            n_examples=3):
    """挑几个句子画出注意力权重分布，看模型把权重压在哪些词上。"""
    import matplotlib.pyplot as plt
    model.eval()
    fig, axes = plt.subplots(n_examples, 1,
                             figsize=(9, 2.4 * n_examples))
    if n_examples == 1:
        axes = [axes]
    for ax, text in zip(axes, texts[:n_examples]):
        toks = list(__import__("jieba").cut(text))[:max_len]
        ids = torch.tensor([[vocab.get(w, UNK) for w in toks]], dtype=torch.long)
        with torch.no_grad():
            logits, w = model(ids, return_attn=True)
            pred = int(logits.argmax(-1))
        w = w[0, :len(toks)].numpy()
        ax.bar(range(len(toks)), w, color="#4C78A8")
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("注意力权重")
        ax.set_title(f"预测：{'正面' if pred == 1 else '负面'}   |   {text[:26]}…",
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {save_path}")


def plot_curves(hists, save_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in hists.items():
        axes[0].plot(h["train_loss"], "o-", label=name, markersize=4)
        axes[1].plot(h["val_acc"], "s-", label=name, markersize=4)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("训练损失")
    axes[0].set_title("训练损失曲线")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("验证准确率")
    axes[1].set_title("验证准确率曲线")
    for ax in axes:
        ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="训练样本上限")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    set_seed(42)

    print("=" * 60)
    print("数据准备：ChnSentiCorp 中文情感二分类")
    tr_x, tr_y, te_x, te_y = load_chn_split(limit=args.limit)
    print(f"   train={len(tr_x)}  test={len(te_x)}")

    # 从训练集里再切 10% 做验证集（指导书要求有验证集）
    n_val = max(200, int(len(tr_x) * 0.1))
    va_x, va_y = tr_x[:n_val], tr_y[:n_val]
    tr_x, tr_y = tr_x[n_val:], tr_y[n_val:]
    print(f"   切分后 train={len(tr_x)}  val={len(va_x)}  test={len(te_x)}")

    print("   分词中（jieba）…")
    tr_tok = zh_tokenize(tr_x)
    va_tok = zh_tokenize(va_x)
    te_tok = zh_tokenize(te_x)
    vocab = build_vocab(tr_tok, min_freq=2)
    print(f"   词表大小={len(vocab) + 2}（含 pad/unk）")

    tr_loader = make_loader(encode(tr_tok, vocab, args.max_len), tr_y,
                            args.batch, shuffle=True)
    va_loader = make_loader(encode(va_tok, vocab, args.max_len), va_y, args.batch)
    te_loader = make_loader(encode(te_tok, vocab, args.max_len), te_y, args.batch)

    v = len(vocab) + 2
    models = {
        "LSTM": LSTMClassifier(v),
        "LSTM+Attention": LSTMAttentionClassifier(v),
        "Transformer": TransformerClassifier(v),
    }

    rows, hists = [], {}
    for name, model in models.items():
        print("=" * 60)
        print(f"训练 {name}（参数量 {count_parameters(model):,}）")
        hist = train_model(model, tr_loader, va_loader, epochs=args.epochs,
                           lr=args.lr, tag=name)
        acc, prec, rec, f1 = evaluate(model, te_loader)
        print(f"   >>> {name}: 准确率={acc:.4f} 精确率={prec:.4f} "
              f"召回={rec:.4f} F1={f1:.4f} 用时={hist['seconds']:.1f}s")
        rows.append([name, f"{count_parameters(model):,}", f"{acc:.4f}",
                     f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}",
                     f"{hist['seconds']:.1f}"])
        hists[name] = hist
        torch.save(model.state_dict(), os.path.join(OUT, f"cls_{name}.pth"))

    from common.utils import save_comparison_table
    save_comparison_table(
        rows, ["模型", "参数量", "准确率", "精确率", "召回率", "F1", "用时(s)"],
        os.path.join(OUT, "exp5_cls_compare.png"),
        "实验五 三种文本分类模型对比（ChnSentiCorp 测试集）")
    plot_curves(hists, os.path.join(OUT, "exp5_cls_curves.png"))

    # 注意力可视化：挑两句带转折的，看模型有没有抓到真正表意的词
    demo = ["环境不错，但是服务太差了，不会再来了",
            "房间干净整洁，前台态度也很好，非常满意"]
    demo = [t for t in demo if t]
    plot_attention_examples(models["LSTM+Attention"], demo, vocab,
                            args.max_len,
                            os.path.join(OUT, "exp5_cls_attention.png"))

    print("=" * 60)
    print(f"结果已保存到 {OUT}")
    print("BERT 的结果见 bert_finetune.py，两者合起来就是指导书要求的四模型对比。")


if __name__ == "__main__":
    main()
