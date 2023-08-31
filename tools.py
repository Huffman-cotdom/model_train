import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from logging_utils import logger
from seqeval.metrics import accuracy_score as seqeval_accuracy_score
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2
from sklearn.metrics import accuracy_score, classification_report
from torch import nn


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def multi_label_classification_metrics(label2id, y_pred, y_true, average="weighted"):
    """
    多标签分类任务的评价指标
    :param y_true: 真实标签，shape为(N, C)
    :param y_pred: 预测标签，shape为(N, C)
    :param average: 按哪种方式计算评价指标，可选'micro'、'macro'、'weighted'
    :return: 准确率（accuracy）、精确率（precision）、召回率（recall）、F1值（f1_score）
    """
    accuracy = accuracy_score(y_true, y_pred)
    CFR = classification_report(
        y_true,
        y_pred,
        labels=list(label2id.values()),
        target_names=list(label2id.keys()),
        zero_division=0,
        digits=4,
        output_dict=True,
    )
    precision = float(CFR[average + " avg"]["precision"])
    recall = float(CFR[average + " avg"]["recall"])
    f1 = float(CFR[average + " avg"]["f1-score"])
    if average not in ["micro", "macro", "weighted"]:
        raise ValueError("average parameter should be 'micro', 'macro', or 'weighted'.")
    return accuracy, precision, recall, f1, CFR


def token_label_classification_metrics(
    token_id2label, y_preds, y_trues, average="weighted"
):
    """
    实体词任务的评价指标
    :param y_true: 真实标签，shape为(N, C)
    :param y_pred: 预测标签，shape为(N, C)
    :param average: 按哪种方式计算评价指标，可选'micro'、'macro'、'weighted'
    :return: 准确率（accuracy）、精确率（precision）、召回率（recall）、F1值（f1_score）
    """
    pred_labels = [
        [token_id2label[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(y_preds, y_trues)
    ]
    true_labels = [
        [token_id2label[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(y_preds, y_trues)
    ]
    accuracy = seqeval_accuracy_score(true_labels, pred_labels)
    CFR = seqeval_classification_report(
        true_labels,
        pred_labels,
        mode="strict",
        zero_division=0,
        digits=4,
        output_dict=True,
        scheme=IOB2,
    )
    precision = float(CFR[average + " avg"]["precision"])
    recall = float(CFR[average + " avg"]["recall"])
    f1 = float(CFR[average + " avg"]["f1-score"])
    if average not in ["micro", "macro", "weighted"]:
        raise ValueError("average parameter should be 'micro', 'macro', or 'weighted'.")
    return accuracy, precision, recall, f1, CFR


if __name__ == "__main__":
    y_pred = [[0, 0, 3, 3], [0, 0, 1, 2]]
    y_true = [[0, 0, 3, 4], [0, 0, 1, 2]]
    token_id2label = {0: "O", 1: "B-LOC", 2: "I-LOC", 3: "B-PER", 4: "I-PER"}
    res = token_label_classification_metrics(token_id2label, y_pred, y_true)
    print(res)
