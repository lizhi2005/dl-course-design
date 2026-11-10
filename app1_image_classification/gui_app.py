# -*- coding: utf-8 -*-
"""综合应用一：智能图像分类系统 —— PyQt5 交互界面。

功能：选择本地图片 -> 模型推理 -> 显示类别、置信度与 Top-3 预测。

运行（需先完成训练产出 app1_best.pth）：
    python app1_image_classification/gui_app.py
"""
import os
import sys
import json

import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit,
                             QProgressBar)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app1_image_classification.dataset import get_transforms  # noqa: E402
from app1_image_classification.train import build_model  # noqa: E402
from PIL import Image  # noqa: E402

CKPT = os.path.join(ROOT, "app1_image_classification", "app1_best.pth")
CLASSES_JSON = os.path.join(ROOT, "app1_image_classification", "classes.json")


class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能图像分类系统 —— 深度学习综合设计")
        self.resize(680, 600)
        self.model = None
        self.classes = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_ui()
        self._load_model()

    def _init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("智能图像分类系统")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.img_label = QLabel("尚未选择图片")
        self.img_label.setFixedHeight(280)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet(
            "border:1px solid #999; background:#f5f5f5; color:#666;")
        layout.addWidget(self.img_label)

        btn_row = QHBoxLayout()
        self.btn_choose = QPushButton("选择图片")
        self.btn_choose.clicked.connect(self.choose_image)
        self.btn_predict = QPushButton("开始识别")
        self.btn_predict.clicked.connect(self.predict)
        self.btn_predict.setEnabled(False)
        btn_row.addWidget(self.btn_choose)
        btn_row.addWidget(self.btn_predict)
        layout.addLayout(btn_row)

        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setFixedHeight(140)
        self.result.setPlaceholderText("识别结果将显示在这里")
        layout.addWidget(self.result)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        layout.addWidget(self.bar)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#888;")
        layout.addWidget(self.status)

        self.setLayout(layout)
        self.current_path = None

    def _load_model(self):
        if not os.path.exists(CKPT):
            self.status.setText("未找到训练好的模型，请先运行 train.py 完成训练")
            return
        try:
            ck = torch.load(CKPT, map_location="cpu")
            self.classes = ck.get("classes", [])
            if not self.classes and os.path.exists(CLASSES_JSON):
                self.classes = json.load(open(CLASSES_JSON, encoding="utf-8"))
            name = ck.get("model", "mobilenet_v3_small")
            self.model = build_model(name, len(self.classes), pretrained=False)
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval()
            self.status.setText(f"模型已加载：{name}，类别数={len(self.classes)}")
        except Exception as e:
            self.status.setText(f"模型加载失败：{e}")

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.png *.jpeg *.bmp)")
        if not path:
            return
        self.current_path = path
        pix = QPixmap(path).scaled(380, 260, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
        self.img_label.setPixmap(pix)
        self.btn_predict.setEnabled(True)
        self.result.clear()

    def predict(self):
        if self.model is None or not self.current_path:
            return
        try:
            img = Image.open(self.current_path).convert("RGB")
            x = get_transforms(train=False)(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                prob = F.softmax(logits, dim=1)[0]
            top3 = torch.topk(prob, k=min(3, len(prob)))
            lines = [f"识别结果：{self.classes[top3.indices[0].item()]}",
                     f"置信度：{top3.values[0].item()*100:.2f}%", "",
                     "Top-3 预测："]
            for v, i in zip(top3.values, top3.indices):
                lines.append(f"   {self.classes[i.item()]:<12} "
                             f"{v.item()*100:6.2f}%")
            self.result.setText("\n".join(lines))
        except Exception as e:
            self.result.setText(f"推理失败：{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ImageClassifierApp()
    w.show()
    sys.exit(app.exec_())
