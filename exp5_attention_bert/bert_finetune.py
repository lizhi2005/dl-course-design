# -*- coding: utf-8 -*-
"""实验五 任务3~4：BERT 中文文本分类微调，并与 TF-IDF+逻辑回归、LSTM 对比。

数据源优先级：ChnSentiCorp（酒店评论）> 内置中文样例数据（离线兜底）。

CPU 训练提示：默认只取部分样本（--limit），max_length=64。
    实测参考：1000 条样本 × 2 epoch 约需 10—20 分钟。

运行：
    python exp5_attention_bert/bert_finetune.py --epochs 2 --limit 1000
    python exp5_attention_bert/bert_finetune.py --epochs 1 --limit 200  # 冒烟
"""
import os
import sys
import argparse
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import set_seed, save_comparison_table  # noqa: E402
from common.nlp_utils import load_chn_split  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp5_attention_bert", "results")
BERT_DIR = os.path.join(ROOT, "models", "bert-base-chinese")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# ------------------------------------------------------------ 三种方法
def run_tfidf_lr(train_texts, train_labels, test_texts, test_labels):
    """传统方法：TF-IDF + 逻辑回归。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    import jieba

    def cut(texts):
        return [" ".join(jieba.lcut(t)) for t in texts]

    t0 = time.time()
    vec = TfidfVectorizer(max_features=5000)
    Xtr = vec.fit_transform(cut(train_texts))
    Xte = vec.transform(cut(test_texts))
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, train_labels)
    pred = clf.predict(Xte)
    return (accuracy_score(test_labels, pred),
            f1_score(test_labels, pred), time.time() - t0)


def run_bert(train_texts, train_labels, test_texts, test_labels, epochs=2,
             max_length=64, batch_size=8, lr=2e-5):
    """BERT 微调。"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_path = BERT_DIR if os.path.exists(BERT_DIR) else "bert-base-chinese"
    print(f"   加载模型: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2)

    enc_tr = tok(list(train_texts), truncation=True, padding="max_length",
                 max_length=max_length, return_tensors="pt")
    enc_te = tok(list(test_texts), truncation=True, padding="max_length",
                 max_length=max_length, return_tensors="pt")
    y_tr = torch.tensor(list(train_labels), dtype=torch.long)
    y_te = torch.tensor(list(test_labels), dtype=torch.long)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(enc_tr["input_ids"],
                                       enc_tr["attention_mask"], y_tr),
        batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    for ep in range(1, epochs + 1):
        total = 0.0
        for i, (ids, mask, y) in enumerate(loader, 1):
            out = model(input_ids=ids, attention_mask=mask, labels=y)
            loss = out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            if i % 20 == 0:
                print(f"      epoch {ep} batch {i}/{len(loader)} "
                      f"loss={total/i:.4f}  ({time.time()-t0:.0f}s)")
    train_time = time.time() - t0

    # 评估
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(y_te), batch_size):
            ids = enc_te["input_ids"][i:i + batch_size]
            mask = enc_te["attention_mask"][i:i + batch_size]
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds.append(logits.argmax(-1))
    pred = torch.cat(preds)
    acc = (pred == y_te).float().mean().item()
    tp = ((pred == 1) & (y_te == 1)).sum().item()
    fp = ((pred == 1) & (y_te == 0)).sum().item()
    fn = ((pred == 0) & (y_te == 1)).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, "bert_finetune.pth"))
    return acc, f1, train_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=1000,
                    help="训练样本上限（CPU 建议 1000 以内）")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-bert", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    set_seed(42)

    print("=" * 60)
    print("数据准备：中文情感二分类")
    train_texts, train_labels, test_texts, test_labels = load_chn_split(
        limit=args.limit)
    print(f"   训练集 {len(train_texts)} 条，测试集 {len(test_texts)} 条")

    rows = []
    print("=" * 60)
    print("方法 1：TF-IDF + 逻辑回归（传统机器学习基线）")
    acc, f1, t = run_tfidf_lr(train_texts, train_labels, test_texts, test_labels)
    print(f"   准确率={acc:.4f}  F1={f1:.4f}  用时={t:.1f}s")
    rows.append(["TF-IDF + 逻辑回归", f"{acc:.4f}", f"{f1:.4f}", f"{t:.1f}"])

    if not args.skip_bert:
        print("=" * 60)
        print("方法 2：BERT 微调（bert-base-chinese）")
        acc, f1, t = run_bert(train_texts, train_labels, test_texts,
                              test_labels, epochs=args.epochs,
                              max_length=args.max_length,
                              batch_size=args.batch)
        print(f"   准确率={acc:.4f}  F1={f1:.4f}  用时={t:.1f}s")
        rows.append(["BERT 微调", f"{acc:.4f}", f"{f1:.4f}", f"{t:.1f}"])

    save_comparison_table(rows, ["方法", "准确率", "F1", "用时(s)"],
                          os.path.join(OUT, "exp5_table_bert.png"),
                          "实验五 中文文本分类方法对比")
    print("=" * 60)
    print("实验三的 LSTM 结果请一并填入对比表（见 exp4_rnn/imdb_lstm.py 输出）。")


if __name__ == "__main__":
    main()
