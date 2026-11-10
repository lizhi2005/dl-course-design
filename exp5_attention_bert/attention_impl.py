# -*- coding: utf-8 -*-
"""实验五 任务1：从零实现缩放点积注意力与加性注意力，并完成 toy seq2seq 翻译。

任务：把英文数字单词序列翻译为数字序列，且目标顺序与源顺序相反。
      （倒序使得注意力热力图呈现"反对角线"，可直观验证模型学会了对齐）

运行：
    python exp5_attention_bert/attention_impl.py --epochs 20
"""
import os
import sys
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import set_seed  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp5_attention_bert", "results")


# ------------------------------------------------------- 1) 两种注意力打分
def scaled_dot_product_attention(q, k, v, mask=None):
    """缩放点积注意力。

        score = q·k^T / sqrt(d)
        attn  = softmax(score)
        out   = attn @ v
    """
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class AdditiveAttention(nn.Module):
    """加性（Bahdanau）注意力：score = v^T tanh(W1 q + W2 k)"""

    def __init__(self, hidden):
        super().__init__()
        self.W1 = nn.Linear(hidden, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self, query, values, mask=None):
        # query: (N, 1, d)  values: (N, T, d)
        scores = self.v(torch.tanh(self.W1(query) + self.W2(values)))
        scores = scores.squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        return context, weights


def check_attention():
    """验证手写注意力与 torch 内置实现一致。"""
    set_seed(42)
    N, Tq, Tk, d = 2, 3, 5, 8
    q, k, v = torch.randn(N, Tq, d), torch.randn(N, Tk, d), torch.randn(N, Tk, d)
    mine, w_mine = scaled_dot_product_attention(q, k, v)
    if hasattr(F, "scaled_dot_product_attention"):
        ref = F.scaled_dot_product_attention(q, k, v)   # 返回输出张量
    else:
        ref = w_mine @ v
    print(f"缩放点积注意力 手写 vs torch 最大误差: "
          f"{(mine - ref).abs().max():.2e}")
    print(f"   注意力权重每行和为 1: "
          f"{bool(torch.allclose(w_mine.sum(-1), torch.ones(N, Tq), atol=1e-5))}")

    att = AdditiveAttention(d)
    ctx, w = att(q[:, :1, :], v)
    print(f"加性注意力 上下文向量形状={tuple(ctx.shape)} "
          f"权重形状={tuple(w.shape)}（应为 (N, T)）")


# ------------------------------------------------------- 2) toy 翻译数据
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine"]
SOS, EOS, PAD = "<sos>", "<eos>", "<pad>"


def make_toy_data(n=3000, min_len=3, max_len=6, seed=42):
    """源：英文数字单词序列；目标：倒序的数字序列。"""
    rnd = random.Random(seed)
    pairs = []
    for _ in range(n):
        L = rnd.randint(min_len, max_len)
        nums = [rnd.randrange(10) for _ in range(L)]
        src = [WORDS[d] for d in nums]
        tgt = [str(d) for d in nums][::-1]      # 倒序，便于观察对齐
        pairs.append((src, tgt))
    return pairs


class ToyVocab:
    def __init__(self, pairs):
        self.src_w2i = {PAD: 0, SOS: 1, EOS: 2}
        self.tgt_w2i = {PAD: 0, SOS: 1, EOS: 2}
        for s, t in pairs:
            for w in s:
                self.src_w2i.setdefault(w, len(self.src_w2i))
            for w in t:
                self.tgt_w2i.setdefault(w, len(self.tgt_w2i))
        self.src_i2w = {i: w for w, i in self.src_w2i.items()}
        self.tgt_i2w = {i: w for w, i in self.tgt_w2i.items()}

    def encode_src(self, seq):
        return [self.src_w2i[w] for w in seq]

    def encode_tgt(self, seq):
        return [self.tgt_w2i[SOS]] + [self.tgt_w2i[w] for w in seq] + \
               [self.tgt_w2i[EOS]]


def pad_batch(seqs, pad_id=0):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [pad_id] * (L - len(s)) for s in seqs],
                        dtype=torch.long)


# ------------------------------------------------------- 3) Seq2Seq + Attention
class Seq2SeqAttention(nn.Module):
    """编码器 LSTM + 带加性注意力的解码器。"""

    def __init__(self, src_vocab, tgt_vocab, embed_dim=64, hidden=128):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab
        self.enc_embed = nn.Embedding(src_vocab, embed_dim, padding_idx=0)
        self.dec_embed = nn.Embedding(tgt_vocab, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.attention = AdditiveAttention(hidden)
        self.decoder = nn.LSTM(embed_dim + hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, tgt_vocab)

    def encode(self, src):
        e = self.enc_embed(src)
        out, (h, c) = self.encoder(e)
        return out, (h, c)

    def forward(self, src, tgt_in, src_mask=None):
        enc_out, (h, c) = self.encode(src)
        e = self.dec_embed(tgt_in)
        T = tgt_in.size(1)
        logits = []
        attn_weights = []
        for t in range(T):
            query = h[-1].unsqueeze(1)                    # (N, 1, H)
            context, w = self.attention(query, enc_out, src_mask)
            dec_in = torch.cat([e[:, t, :], context], dim=-1).unsqueeze(1)
            out, (h, c) = self.decoder(dec_in, (h, c))
            logits.append(self.fc(out.squeeze(1)))
            attn_weights.append(w)
        return torch.stack(logits, dim=1), torch.stack(attn_weights, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src, max_len=10, sos_id=1, eos_id=2):
        self.eval()
        enc_out, (h, c) = self.encode(src)
        token = torch.tensor([[sos_id]] * src.size(0), dtype=torch.long)
        outputs, weights = [], []
        for _ in range(max_len):
            e = self.dec_embed(token)
            query = h[-1].unsqueeze(1)
            context, w = self.attention(query, enc_out)
            dec_in = torch.cat([e, context.unsqueeze(1)], dim=-1)
            out, (h, c) = self.decoder(dec_in, (h, c))
            pred = self.fc(out.squeeze(1)).argmax(-1)
            outputs.append(pred)
            weights.append(w)
            token = pred.unsqueeze(1)
        return torch.stack(outputs, dim=1), torch.stack(weights, dim=1)


def train_seq2seq(model, pairs, vocab, epochs=20, lr=1e-3, batch_size=64):
    set_seed(42)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    losses = []
    for ep in range(1, epochs + 1):
        random.shuffle(pairs)
        total, cnt = 0.0, 0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            src = pad_batch([vocab.encode_src(s) for s, _ in batch])
            tgt = pad_batch([vocab.encode_tgt(t) for _, t in batch])
            src_mask = (src != 0)
            logits, _ = model(src, tgt[:, :-1], src_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                           tgt[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); cnt += 1
        losses.append(total / cnt)
        if ep % 5 == 0 or ep == 1:
            print(f"   epoch {ep:>2d}/{epochs}  loss={total/cnt:.4f}")
    return losses


def plot_attention_heatmap(weights, src_tokens, tgt_tokens, save_path=None):
    """绘制注意力权重热力图。"""
    w = weights[:len(tgt_tokens), :len(src_tokens)]
    fig, ax = plt.subplots(figsize=(0.55 * len(src_tokens) + 2,
                                    0.5 * len(tgt_tokens) + 2))
    im = ax.imshow(w, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens)
    ax.set_xlabel("源序列（编码器时间步）")
    ax.set_ylabel("生成的目标词")
    ax.set_title("注意力权重热力图（颜色越亮表示关注越强）")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"已保存: {save_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("=" * 60)
    print("任务1：两种注意力打分函数实现与验证")
    check_attention()

    print("=" * 60)
    print("任务2：toy 序列翻译（英文数字 -> 倒序数字）")
    pairs = make_toy_data(n=args.n)
    vocab = ToyVocab(pairs)
    print(f"   样本数={len(pairs)}  源词表={len(vocab.src_w2i)} "
          f"目标词表={len(vocab.tgt_w2i)}")
    print(f"   示例：{' '.join(pairs[0][0])} -> {' '.join(pairs[0][1])}")

    model = Seq2SeqAttention(len(vocab.src_w2i), len(vocab.tgt_w2i))
    losses = train_seq2seq(model, pairs, vocab, epochs=args.epochs)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(losses, "o-")
    ax.set_xlabel("Epoch"); ax.set_ylabel("损失")
    ax.set_title("实验五 Seq2Seq+Attention 训练损失")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp5_seq2seq_loss.png"), dpi=150)
    plt.close(fig)

    print("=" * 60)
    print("任务3：贪心解码示例与注意力可视化")
    src_text = ["three", "one", "seven", "two"]
    src = pad_batch([vocab.encode_src(src_text)])
    preds, weights = model.greedy_decode(src, max_len=len(src_text) + 2)
    pred_tokens = [vocab.tgt_i2w[int(i)] for i in preds[0]
                   if int(i) not in (0, 1, 2)]
    print(f"   源: {' '.join(src_text)}")
    print(f"   目标(应倒序): {' '.join(str(WORDS.index(w)) for w in src_text[::-1])}")
    print(f"   模型输出: {' '.join(pred_tokens)}")

    plot_attention_heatmap(weights[0].numpy(), src_text, pred_tokens,
                           os.path.join(OUT, "exp5_attention_heatmap.png"))


if __name__ == "__main__":
    main()
