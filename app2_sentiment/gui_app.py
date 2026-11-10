# -*- coding: utf-8 -*-
"""综合应用二：智能文本情感分析系统 —— PyQt5 交互界面。

功能：输入中文文本 -> BERT 模型推理 -> 显示情感倾向（正面/负面）与置信度。

运行（需先完成训练产出 app2_sentiment/model）：
    python app2_sentiment/gui_app.py
"""
import os
import sys
import json

import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QTextEdit, QProgressBar)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SAVE_DIR = os.path.join(ROOT, "app2_sentiment", "model")


class SentimentApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能文本情感分析系统 —— 深度学习综合设计")
        self.resize(620, 480)
        self.model, self.tok = None, None
        self._init_ui()
        self._load_model()

    def _init_ui(self):
        layout = QVBoxLayout()
        title = QLabel("智能文本情感分析系统")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addWidget(QLabel("请输入待分析的文本："))
        self.input = QTextEdit()
        self.input.setFixedHeight(130)
        self.input.setPlaceholderText("例如：这家酒店环境很好，服务也很周到")
        layout.addWidget(self.input)

        row = QHBoxLayout()
        self.btn = QPushButton("开始分析")
        self.btn.clicked.connect(self.analyze)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(lambda: self.input.clear())
        row.addWidget(self.btn)
        row.addWidget(self.btn_clear)
        layout.addLayout(row)

        self.result_label = QLabel("等待输入")
        self.result_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "padding:12px; border:1px solid #ccc; background:#fafafa;")
        layout.addWidget(self.result_label)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setFixedHeight(110)
        layout.addWidget(self.detail)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        layout.addWidget(self.bar)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#888;")
        layout.addWidget(self.status)
        self.setLayout(layout)

    def _load_model(self):
        if not os.path.exists(SAVE_DIR):
            self.status.setText("未找到模型，请先运行 app2_sentiment/train.py")
            return
        try:
            from transformers import AutoTokenizer, \
                AutoModelForSequenceClassification
            self.tok = AutoTokenizer.from_pretrained(SAVE_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                SAVE_DIR)
            self.model.eval()
            meta_path = os.path.join(SAVE_DIR, "meta.json")
            extra = ""
            if os.path.exists(meta_path):
                meta = json.load(open(meta_path, encoding="utf-8"))
                extra = f"  测试准确率={meta.get('acc', 0):.4f}"
            self.status.setText(f"模型已加载（BERT 微调）{extra}")
        except Exception as e:
            self.status.setText(f"模型加载失败：{e}")

    def analyze(self):
        text = self.input.toPlainText().strip()
        if not text:
            self.result_label.setText("请输入文本")
            return
        if self.model is None:
            self.result_label.setText("模型未加载")
            return
        try:
            enc = self.tok(text, truncation=True, padding="max_length",
                           max_length=64, return_tensors="pt")
            with torch.no_grad():
                prob = F.softmax(self.model(**enc).logits, dim=1)[0]
            neg, pos = prob[0].item(), prob[1].item()
            label = "正面" if pos >= neg else "负面"
            color = "#c0392b" if label == "正面" else "#27ae60"
            self.result_label.setText(f"情感倾向：{label}    置信度："
                                      f"{max(pos, neg)*100:.2f}%")
            self.result_label.setStyleSheet(
                f"padding:12px; border:1px solid #ccc; background:#fafafa; "
                f"color:{color};")
            self.detail.setText(
                f"正面概率：{pos*100:.2f}%\n负面概率：{neg*100:.2f}%\n\n"
                f"文本长度：{len(text)} 字")
        except Exception as e:
            self.result_label.setText(f"分析失败：{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SentimentApp()
    w.show()
    sys.exit(app.exec_())
