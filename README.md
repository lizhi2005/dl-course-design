# 深度学习综合设计（课程设计）

《深度学习综合设计》课程的全部代码：**5 个基础实验 + 2 个综合应用**，全部在 CPU 环境可复现。

## 项目内容

| 目录 | 内容 |
|---|---|
| `exp1_mlp` | 多层感知机：手写实现 + PyTorch 对比；激活函数（Sigmoid/Tanh/ReLU）、学习率、隐藏层规模、批量大小超参对比 |
| `exp2_cnn` | 卷积神经网络：手写卷积前向传播与 PyTorch 逐元素对拍；MNIST 与 CIFAR-10 分类 |
| `exp3_cnn_advanced` | CNN 进阶：BN / Dropout / 数据增强正则化对照实验；ResNet18 迁移学习（特征提取 vs 从头训练） |
| `exp4_rnn` | 循环神经网络：手写 RNN / LSTM 单元对拍；LSTM 中文文本分类 |
| `exp5_attention_bert` | 注意力机制：LSTM / LSTM+Attention / Transformer 中文情感分类对比；BERT 微调 |
| `app1_image_classification` | 应用一：CIFAR-10 智能图像分类系统（Gradio 交互界面） |
| `app2_sentiment` | 应用二：中文智能文本情感分析系统（PyQt5 交互界面，LSTM 与 BERT 双模型可切换） |
| `common` | 公共工具（数据加载、训练循环等） |
| `logs` | 部分训练日志 |

## 亮点结果

- ResNet18 **迁移学习**在 CIFAR-10 子集上达 **95.65%**，而从头训练仅 67.39%（第 1 轮即达 90%+）
- 应用二：BERT 微调准确率 **94.75%**，LSTM（加 `pack_padded_sequence` 修正）**86.50%**
- 结构化改进（packing +8.2 点、BN +9.7 点、注意力 +10 点）收益远大于单纯调超参

## 环境

```
Python 3.9
PyTorch 2.x（CPU 即可）
transformers、datasets、jieba、gradio、PyQt5
```

## 运行示例

```bash
# 实验一：MLP 基线 + 超参对比
python exp1_mlp/mlp_torch.py
python exp1_mlp/activation_compare.py

# 实验五：BERT 微调
python exp5_attention_bert/bert_finetune.py

# 应用二：PyQt5 情感分析界面
python app2_sentiment/gui_app.py
```

> 数据集（MNIST / CIFAR-10 / ChnSentiCorp）与预训练模型（bert-base-chinese）运行时自动下载或由脚本获取，未随仓库上传。
