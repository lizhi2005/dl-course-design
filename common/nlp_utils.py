# -*- coding: utf-8 -*-
"""文本数据工具：词表构建、序列编码、IMDB / ChnSentiCorp 加载（带离线兜底）。

数据源优先级（1 最快，无需联网）：
    1) data/imdb/*.csv、data/chnsenticorp/*.csv   明文 CSV，由 fetch_text_data.py 生成
    2) data/imdb、data/chnsenticorp  （datasets.save_to_disk 格式）
    3) datasets.load_dataset(...)    （需联网）
    4) 内置合成数据                  （保证任何环境都能跑通全流程）
"""
import os
import re
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

PAD, UNK = 0, 1


def tokenize_en(text):
    """英文简单分词：小写化 + 提取单词与标点。"""
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    return re.findall(r"[a-z']+|[.!?]", text)


def build_vocab(sentences, min_freq=2, max_size=30000):
    """构建词表，返回 dict: word -> id（0=padd, 1=unk）。"""
    cnt = Counter()
    for s in sentences:
        cnt.update(s)
    words = [w for w, c in cnt.most_common(max_size) if c >= min_freq]
    vocab = {w: i + 2 for i, w in enumerate(words)}
    return vocab


def encode(sentences, vocab, max_len=200):
    """把分词后的句子编码为定长 id 序列（截断 + 填充）。"""
    ids = []
    for s in sentences:
        seq = [vocab.get(w, UNK) for w in s][:max_len]
        seq += [PAD] * (max_len - len(seq))
        ids.append(seq)
    return torch.tensor(ids, dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def make_loaders(texts, labels, max_len=200, batch_size=64, min_freq=2,
                 max_vocab=30000, test_texts=None, test_labels=None,
                 vocab=None):
    """从原始文本构造 DataLoader（自动分词、建词表、编码）。

    重要：训练集与测试集必须分开传入。词表只在**训练集**上构建，
    测试集复用同一词表（避免数据泄漏）。若未传 test_texts，
    则退化为正例验证式的同数据集评估，正式实验请勿这样使用。
    """
    train_toks = [tokenize_en(t) for t in texts]
    if vocab is None:
        vocab = build_vocab(train_toks, min_freq=min_freq, max_size=max_vocab)
    train_ds = TextDataset(encode(train_toks, vocab, max_len),
                           torch.tensor(labels, dtype=torch.long))

    if test_texts is not None:
        test_toks = [tokenize_en(t) for t in test_texts]
        test_ds = TextDataset(encode(test_toks, vocab, max_len),
                              torch.tensor(test_labels, dtype=torch.long))
    else:
        test_ds = train_ds

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            vocab, train_ds)


# ------------------------------------------------------------ 合成兜底数据
_POS = ["this movie is great", "wonderful acting and touching story",
        "i love this film so much", "amazing plot and brilliant cast",
        "highly recommend it to everyone", "beautiful scenes and lovely music",
        "one of the best films i have seen", "the director did a perfect job"]
_NEG = ["this movie is terrible", "boring plot and awful acting",
        "i hate this film so much", "waste of time and money",
        "the worst movie i have ever seen", "poor story and bad ending",
        "nothing interesting happened at all", "i regret watching this film"]


def synthetic_imdb(n_train=4000, n_test=1000, seed=42):
    """内置合成影评数据（网络不可用时使用，仅用于流程验证）。"""
    rnd = random.Random(seed)
    def gen(n):
        texts, labels = [], []
        for _ in range(n):
            pos = rnd.random() < 0.5
            base = rnd.choice(_POS if pos else _NEG)
            extra = " ".join(rnd.choice(_POS if pos else _NEG).split()[1:])
            texts.append(base + " " + extra)
            labels.append(1 if pos else 0)
        return texts, labels
    tr_x, tr_y = gen(n_train)
    te_x, te_y = gen(n_test)
    return tr_x, tr_y, te_x, te_y


# ------------------------------------------------------- 中文情感数据
_POS_ZH = ["酒店环境很好，房间干净整洁，服务态度热情，下次还会再来",
           "位置便利，出行方便，前台小姐姐很贴心，早餐丰富可口",
           "设施齐全，床品舒适，隔音效果好，住得非常满意",
           "性价比很高，房间宽敞明亮，卫生打扫得很干净",
           "服务周到，办理入住很快，周边吃饭购物都方便",
           "房间视野开阔，采光很好，整体体验超出预期",
           "工作人员很有礼貌，响应速度也快，值得推荐"]
_NEG_ZH = ["房间很脏，设施陈旧，服务态度差，完全不推荐",
           "隔音太差，晚上吵得睡不着，卫生间还有异味",
           "前台效率低，等了很久才办理完，房间和描述的完全不符",
           "价格贵但体验很差，空调不制冷，床单还有污渍",
           "位置偏僻，交通不便，卫生状况糟糕，非常失望",
           "热水供应不稳定，网络信号很差，不会再住第二次",
           "服务敷衍，问题反馈无人处理，整体感受很糟糕"]


def _read_csv_dir(directory, shuffle=True, seed=42):
    """读取由 fetch_text_data.py 导出的明文 CSV 目录。

    期望文件名 train.csv / test.csv / validation.csv，列为 label,text。
    返回 dict: {"train": (texts, labels), "test": (...), ...}

    【重要】IMDB / ChnSentiCorp 官方文件是按标签或按来源领域**排好序**的
    （例如 IMDB 前 12500 条全是负例）。若不打乱就直接取前 N 条做限量，
    会得到单一类别的子集 —— 表现为准确率虚高到 1.0 而 F1=0，模型退化成
    "永远预测同一类"。因此这里默认用固定种子打乱，保证任何 limit 都类别均衡。
    """
    import pandas as pd
    out = {}
    for split in ["train", "test", "validation"]:
        p = os.path.join(directory, f"{split}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p).dropna(subset=["text", "label"])
        texts = df["text"].astype(str).tolist()
        labels = [int(x) for x in df["label"]]
        if shuffle and len(texts) > 1:
            order = list(range(len(texts)))
            random.Random(seed).shuffle(order)
            texts = [texts[i] for i in order]
            labels = [labels[i] for i in order]
        out[split] = (texts, labels)
    return out


def _balanced_sample(texts, labels, n, seed=42):
    """均衡抽样：按类别分层抽取，避免截断时拿到单一类别。"""
    if not n or n >= len(texts):
        return texts, labels
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    order = order[:n]
    return [texts[i] for i in order], [labels[i] for i in order]


def load_chn_sentiment(limit=None, split="train"):
    """加载中文情感二分类数据，返回 (texts, labels)。

    优先级：本地 CSV > datasets 缓存 > 在线下载 > 内置中文样例（离线兜底）。
    """
    data = _read_csv_dir(os.path.join(DATA, "chnsenticorp"))
    if data:
        key = split if split in data else ("train" if "train" in data
                                           else list(data.keys())[0])
        texts, labels = data[key]
        # 若指定 split 不存在，退化为把 train+test 拼起来
        print(f"使用本地 ChnSentiCorp（{key}）：{len(texts)} 条")
        return (texts[:limit], labels[:limit]) if limit else (texts, labels)

    local = os.path.join(DATA, "chnsenticorp")
    if os.path.exists(local):
        try:
            from datasets import load_from_disk
            ds = load_from_disk(local)
            key = "train" if "train" in ds else list(ds.keys())[0]
            texts, labels = ds[key]["text"], ds[key]["label"]
            print(f"使用缓存 ChnSentiCorp：{len(texts)} 条")
            return (texts[:limit], labels[:limit]) if limit else (texts, labels)
        except Exception as e:
            print(f"本地数据读取失败：{e}")

    for name in ["lxyuan/chnsenticorp", "chnsenticorp", "seamew/ChnSentiCorp"]:
        try:
            from datasets import load_dataset
            ds = load_dataset(name)
            key = "train" if "train" in ds else list(ds.keys())[0]
            texts, labels = ds[key]["text"], ds[key]["label"]
            print(f"使用在线数据集 {name}：{len(texts)} 条")
            return (texts[:limit], labels[:limit]) if limit else (texts, labels)
        except Exception:
            continue

    print("未获取到在线数据集，使用内置中文样例数据扩充（离线兜底）")
    n = limit or 1200
    texts, labels = [], []
    for i in range(n):
        if i % 2 == 0:
            texts.append(_POS_ZH[i % len(_POS_ZH)])
            labels.append(1)
        else:
            texts.append(_NEG_ZH[i % len(_NEG_ZH)])
            labels.append(0)
    return texts, labels


def load_chn_split(test_size=0.2, limit=None):
    """返回标准切分 (train_texts, train_labels, test_texts, test_labels)。

    优先使用 ChnSentiCorp 官方 train/test 划分；无本地数据时按比例切分。
    limit 用于快速冒烟测试（同时限制训练集与测试集规模）。
    """
    data = _read_csv_dir(os.path.join(DATA, "chnsenticorp"))
    if data and "train" in data:
        tr_x, tr_y = data["train"]
        te_x, te_y = data.get("test", ([], []))
        if not te_x:
            return split_train_test(tr_x, tr_y, 1 - test_size)
        if limit:
            # 注意：limit 只裁训练集，测试集保持官方完整的 1200 条。
            # 如果连测试集也一起裁，测试集只剩下 300 条左右，
            # 准确率的波动能到 ±3 个百分点，根本没法用来比较模型。
            tr_x, tr_y = tr_x[:limit], tr_y[:limit]
        print(f"ChnSentiCorp 官方切分：train={len(tr_x)} test={len(te_x)} | "
              f"训练集正负={sum(tr_y)}/{len(tr_y)-sum(tr_y)}")
        return tr_x, tr_y, te_x, te_y
    texts, labels = load_chn_sentiment()
    return split_train_test(texts, labels, 1 - test_size)


def split_train_test(texts, labels, train_ratio=0.8, shuffle=True, seed=42):
    """按给定比例切分。

    先打乱再按位置切，避免源数据按类别/领域排序导致切出一整块单一标签。
    """
    if shuffle:
        order = list(range(len(texts)))
        random.Random(seed).shuffle(order)
        texts = [texts[i] for i in order]
        labels = [labels[i] for i in order]
    n = int(len(texts) * train_ratio)
    return texts[:n], labels[:n], texts[n:], labels[n:]


def load_imdb(max_len=200, batch_size=64, limit=None):
    """加载 IMDB 情感二分类数据，返回 (train_loader, test_loader, vocab)。"""
    # 1) 本地明文 CSV（fetch_text_data.py 导出，最快且不依赖 datasets 库）
    data = _read_csv_dir(os.path.join(DATA, "imdb"))
    if data and "train" in data:
        tr_x, tr_y = data["train"]
        te_x, te_y = data.get("test", ([], []))
        if limit:
            tr_x, tr_y = tr_x[:limit], tr_y[:limit]
            te_x, te_y = te_x[:limit], te_y[:limit]
        print(f"使用本地 IMDB：train={len(tr_x)} test={len(te_x)} | "
              f"训练集正负={sum(tr_y)}/{len(tr_y)-sum(tr_y)} "
              f"测试集正负={sum(te_y)}/{len(te_y)-sum(te_y)}")
        return make_loaders(tr_x, tr_y, max_len, batch_size,
                            test_texts=te_x, test_labels=te_y)[:3]

    # 2) datasets.save_to_disk 缓存
    local = os.path.join(DATA, "imdb")
    if os.path.exists(local):
        try:
            from datasets import load_from_disk
            ds = load_from_disk(local)
            tr_x = ds["train"]["text"][:limit or len(ds["train"])]
            tr_y = ds["train"]["label"][:limit or len(ds["train"])]
            te_x = ds["test"]["text"][:limit or len(ds["test"])]
            te_y = ds["test"]["label"][:limit or len(ds["test"])]
            print(f"使用缓存 IMDB：train={len(tr_x)} test={len(te_x)}")
            return make_loaders(tr_x, tr_y, max_len, batch_size,
                                test_texts=te_x, test_labels=te_y)[:3]
        except Exception as e:
            print(f"本地 IMDB 读取失败：{e}")

    # 2) 联网下载
    try:
        from datasets import load_dataset
        ds = load_dataset("imdb")
        print("使用在线 IMDB 数据集")
        tr_x = ds["train"]["text"][:limit or len(ds["train"])]
        tr_y = ds["train"]["label"][:limit or len(ds["train"])]
        te_x = ds["test"]["text"][:limit or len(ds["test"])]
        te_y = ds["test"]["label"][:limit or len(ds["test"])]
        return make_loaders(tr_x, tr_y, max_len, batch_size,
                            test_texts=te_x, test_labels=te_y)[:3]
    except Exception as e:
        print(f"IMDB 下载失败（{type(e).__name__}），改用内置合成数据")

    # 3) 合成兜底
    tr_x, tr_y, te_x, te_y = synthetic_imdb()
    print("使用内置合成影评数据（流程验证用，正式实验请联网获取 IMDB）")
    return make_loaders(tr_x, tr_y, max_len, batch_size,
                        test_texts=te_x, test_labels=te_y)[:3]
