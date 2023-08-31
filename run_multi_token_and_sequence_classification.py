import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from arg import args
from logging_utils import logger, metrics_logging
from modeling import MultiTaskBertForSequenceAndTokenClassification
from tools import (
    multi_label_classification_metrics,
    seed_everything,
    token_label_classification_metrics,
)
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    get_scheduler,
)

from data import MYDATA, get_dataLoader, get_multitask_sequence_and_token_dataLoader

MODEL_CLASSES = {
    "bert": BertForSequenceClassification,
    "multi_task_bert": MultiTaskBertForSequenceAndTokenClassification,
}

dataloader = {
    "bert": get_dataLoader,
    "multi_task_bert": get_multitask_sequence_and_token_dataLoader,
}


def to_device(device, batch_data):
    batch_data["batch_inputs"]["input_ids"].to(device)
    device_batch_inputs = {
        k: v.to(device) for k, v in batch_data["batch_inputs"].items()
    }
    device_labels = torch.tensor(batch_data["labels"]).to(device)
    sentences = batch_data.get("sentences", None)
    device_tokens = batch_data.get("batch_tokens", None)
    device_token_labels = torch.tensor(batch_data["token_labels"]).to(device)
    return {
        "batch_inputs": device_batch_inputs,
        "labels": device_labels,
        "sentences": sentences,
        "batch_tokens": device_tokens,
        "token_labels": device_token_labels,
    }


def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"loss: {0:>7f}")
    finish_step_num = epoch * len(dataloader)

    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = to_device(args.device, batch_data)
        batch_inputs = batch_data["batch_inputs"]
        labels = batch_data["labels"]
        token_labels = batch_data["token_labels"]
        outputs = model(
            **batch_inputs, labels=labels, token_labels=token_labels, return_dict=True
        )
        loss = outputs.total_loss
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f"loss: {total_loss/(finish_step_num + step):>7f}")
        progress_bar.update(1)
    return total_loss


def test_loop(args, dataloader, model, mode="Test"):
    model.eval()
    num = 0
    # label
    all_labels, all_probs = [], []
    # token label
    all_token_labels, all_token_probs = [], []
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            num += 1
            batch_data = to_device(args.device, batch_data)
            batch_inputs = batch_data["batch_inputs"]
            outputs = model(**batch_inputs, return_dict=True)

            # label
            cls_logits = outputs.cls_logits
            labels = batch_data["labels"].cpu().numpy()
            if args.problem_type == "single_label_classification":
                probs = cls_logits.argmax(dim=-1).cpu().numpy().tolist()
                all_probs.extend(probs)
                all_labels.extend(labels)
            elif args.problem_type == "multi_label_classification":
                if args.loss_type == "BCE":
                    probs = torch.sigmoid(cls_logits)
                    probs = (probs > 0.5).int().cpu().numpy()
                elif args.loss_type == "ZLPR":
                    probs = (cls_logits > 0).int().cpu().numpy()
                else:
                    raise Exception("loss_type must be ZLPR or BCE")
                all_labels.append(labels)
                all_probs.append(probs)
            # token label
            token_logits = outputs.token_logits
            token_labels = batch_data["token_labels"].cpu().numpy().tolist()
            token_probs = token_logits.argmax(dim=-1).cpu().numpy().tolist()
            all_token_labels.extend(token_labels)
            all_token_probs.extend(token_probs)

    # label
    if args.problem_type == "multi_label_classification":
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        accuracy, precision, recall, f1, _ = multi_label_classification_metrics(
            args.label2id, all_probs, all_labels, "macro"
        )
    elif args.problem_type == "single_label_classification":
        accuracy, precision, recall, f1, _ = multi_label_classification_metrics(
            args.label2id, all_probs, all_labels, "macro"
        )

    # token label
    # all_token_probs = np.concatenate(all_token_probs, axis=0).tolist()
    # all_token_labels = np.concatenate(all_token_labels, axis=0).tolist()
    (
        token_accuracy,
        token_precision,
        token_recall,
        token_f1,
        _,
    ) = token_label_classification_metrics(
        args.token_id2label, all_token_probs, all_token_labels, "micro"
    )

    metrics = {
        "label": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "token_label": {
            "accuracy": token_accuracy,
            "precision": token_precision,
            "recall": token_recall,
            "f1": token_f1,
        },
    }

    return metrics


def train(args, train_dataset, dev_dataset, model, tokenizer):
    """Train the model"""
    train_shuffle, mode = True, "Test"
    if args.use_r_drop:
        train_shuffle, mode = False, "Train"
    train_dataloader = get_multitask_sequence_and_token_dataLoader(
        train_dataset,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=train_shuffle,
        num_labels=args.num_labels,
        problem_type=args.problem_type,
        use_r_drop=args.use_r_drop,
        mode=mode,
    )
    dev_dataloader = get_multitask_sequence_and_token_dataLoader(
        dev_dataset,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=train_shuffle,
        num_labels=args.num_labels,
        problem_type=args.problem_type,
        use_r_drop=args.use_r_drop,
        mode=mode,
    )
    t_total = len(train_dataloader) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        fused=True,  # 融合，减少GPU内存读写次数
    )
    lr_scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total,
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    logger.info(f"Batch size - {args.batch_size}")
    logger.info(f"Total warmup steps - {args.warmup_steps}")
    logger.info(f"Total training steps - {t_total}")
    logger.info(f"early_stop - {args.early_stop}")
    logger.info(f"loss_type - {args.loss_type}")
    with open(os.path.join(args.output_dir, "args.txt"), "wt") as f:
        f.write(str(args))

    total_loss = 0.0
    best_f1 = 0.0
    early_stop = args.early_stop
    for epoch in range(args.num_train_epochs):
        logger.info(
            f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------"
        )
        total_loss = train_loop(
            args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss
        )
        metrics = test_loop(args, dev_dataloader, model)
        valid_f1 = metrics["label"]["f1"]
        metrics_logging(metrics, logger, mode="Dev")
        if early_stop > 0:
            if valid_f1 > best_f1:
                best_f1 = valid_f1
                logger.info(f"saving new weights to {args.output_dir}...\n")
                save_weight = (
                    f"epoch_{epoch+1}_dev_f1_{(100*valid_f1):0.1f}_weights.bin"
                )
                torch.save(
                    model.state_dict(), os.path.join(args.output_dir, save_weight)
                )
                config_dict = config.to_dict()
                del config_dict["args"]
                with open(os.path.join(args.output_dir, "config.json"), "w") as f:
                    json.dump(
                        config_dict, f, indent=2, ensure_ascii=False, sort_keys=True
                    )
            else:
                early_stop -= 1
                logger.info(f"saving new weights to {args.output_dir}...\n")
                save_weight = (
                    f"epoch_{epoch+1}_dev_f1_{(100*valid_f1):0.1f}_weights.bin"
                )
                torch.save(
                    model.state_dict(), os.path.join(args.output_dir, save_weight)
                )
                config_dict = config.to_dict()
                del config_dict["args"]
                with open(os.path.join(args.output_dir, "config.json"), "w") as f:
                    json.dump(
                        config_dict, f, indent=2, ensure_ascii=False, sort_keys=True
                    )
        else:
            logger.info(f"early_stop = {early_stop}, stop training...")
            break
    logger.info("Done!")


def test(args, test_dataset, model, tokenizer, save_weights: list):
    test_dataloader = get_multitask_sequence_and_token_dataLoader(
        test_dataset,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
        num_labels=args.num_labels,
        problem_type=args.problem_type,
        use_r_drop=args.use_r_drop,
        mode="Test",
    )
    logger.info("***** Running testing *****")
    for save_weight in save_weights:
        logger.info(f"loading weights from {save_weight}...")
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, model)
        metrics_logging(metrics, logger, mode="Test")


def predict(args, test_dataset, model):
    test_dataloader = get_multitask_sequence_and_token_dataLoader(
        test_dataset,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
        num_labels=args.num_labels,
        problem_type=args.problem_type,
        use_r_drop=args.use_r_drop,
        mode="Predict",
    )
    model.eval()
    num = 0
    all_data, all_labels, all_probs = [], [], []
    all_token_labels, all_token_probs = [], []
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader):
            num += 1
            batch_data = to_device(args.device, batch_data)
            batch_inputs = batch_data["batch_inputs"]
            outputs = model(**batch_inputs, return_dict=True)
            # label
            cls_logits = outputs.cls_logits
            labels = batch_data["labels"].cpu().numpy()
            if args.problem_type == "single_label_classification":
                probs = cls_logits.argmax(dim=-1).cpu().numpy().tolist()
                all_probs.extend(probs)
                all_labels.extend(labels)
            elif args.problem_type == "multi_label_classification":
                if args.loss_type == "BCE":
                    probs = torch.sigmoid(cls_logits)
                    probs = (probs > 0.5).int().cpu().numpy()
                elif args.loss_type == "ZLPR":
                    probs = (cls_logits > 0).int().cpu().numpy()
                else:
                    raise Exception("loss_type must be ZLPR or BCE")
                all_labels.append(labels)
                all_probs.append(probs)
            # token label
            token_logits = outputs.token_logits
            token_labels = batch_data["token_labels"].cpu().numpy().tolist()
            token_probs = token_logits.argmax(dim=-1).cpu().numpy().tolist()
            all_token_labels.extend(token_labels)
            all_token_probs.extend(token_probs)

            sentences = batch_data["sentences"]
            for i in range(len(sentences)):
                # label
                if args.problem_type == "multi_label_classification":
                    prob = probs[i]
                    prob = prob.nonzero()[0].tolist()
                    prob = [args.id2label[item] for item in prob]
                    label = labels[i]
                    label = label.nonzero()[0].tolist()
                    label = [args.id2label[item] for item in label]
                elif args.problem_type == "single_label_classification":
                    prob = probs[i]
                    prob = args.id2label[prob]
                    label = labels[i]
                    label = args.id2label[label]
                else:
                    raise Exception(
                        "problem_type must be single_label_classification or multi_label_classification"
                    )
                # token label
                token_prob = token_probs[i]

                token_prob = [
                    args.token_id2label[item]
                    for item, _ in zip(token_prob, sentences[i])
                    if item != -100
                ]
                token_label = token_labels[i]
                token_label = [
                    args.token_id2label[item]
                    for item, _ in zip(token_label, sentences[i])
                    if item != -100
                ]
                data = {
                    "sentence": sentences[i],
                    "true_label": label,
                    "pred_label": prob,
                    "true_token_label": token_label,
                    "pred_token_label": token_prob,
                }
                all_data.append(data)
    # label
    if args.problem_type == "multi_label_classification":
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        accuracy, precision, recall, f1, report = multi_label_classification_metrics(
            args.label2id, all_probs, all_labels, "macro"
        )
    elif args.problem_type == "single_label_classification":
        accuracy, precision, recall, f1, report = multi_label_classification_metrics(
            args.label2id, all_probs, all_labels, "macro"
        )
    # token label
    (
        token_accuracy,
        token_precision,
        token_recall,
        token_f1,
        token_report,
    ) = token_label_classification_metrics(
        args.token_id2label, all_token_probs, all_token_labels, "micro"
    )

    metrics = {
        "label": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "token_label": {
            "accuracy": token_accuracy,
            "precision": token_precision,
            "recall": token_recall,
            "f1": token_f1,
        },
    }

    metrics_logging(metrics, logger, mode="Predict")

    # 将输出结果转化为DataFrame
    # label
    df_report = pd.DataFrame(report).transpose()
    df_report["precision"] = df_report["precision"].apply(lambda x: "{:.2%}".format(x))
    df_report["recall"] = df_report["recall"].apply(lambda x: "{:.2%}".format(x))
    df_report["f1-score"] = df_report["f1-score"].apply(lambda x: "{:.2%}".format(x))
    df_report["support"] = df_report["support"].astype(int)
    df_report.index.name = "label"
    df_report.reset_index(inplace=True)
    # token label
    df_token_report = pd.DataFrame(token_report).transpose()
    df_token_report["precision"] = df_token_report["precision"].apply(
        lambda x: "{:.2%}".format(x)
    )
    df_token_report["recall"] = df_token_report["recall"].apply(
        lambda x: "{:.2%}".format(x)
    )
    df_token_report["f1-score"] = df_token_report["f1-score"].apply(
        lambda x: "{:.2%}".format(x)
    )
    df_token_report["support"] = df_token_report["support"].astype(int)
    df_token_report.index.name = "label"
    df_token_report.reset_index(inplace=True)

    # label 和 token_label 合并
    df_report = pd.concat([df_report, df_token_report], axis=0)

    # 保存结果
    df = pd.DataFrame(
        all_data,
        columns=[
            "sentence",
            "true_label",
            "pred_label",
            "true_token_label",
            "pred_token_label",
        ],
    )
    return df_report, df


if __name__ == "__main__":
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    with open(args.label_file) as f:
        label2id = json.load(f)
    with open(args.token_label_file) as f:
        token_label2id = json.load(f)
    # label
    args.label2id = label2id
    args.id2label = {v: k for k, v in label2id.items()}
    args.num_labels = len(label2id)
    # token label
    args.token_label2id = token_label2id
    args.token_id2label = {v: k for k, v in token_label2id.items()}
    args.num_token_labels = len(token_label2id)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f"Using {args.device} device, n_gpu: {args.n_gpu}")
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    logger.info(f"loading pretrained model and tokenizer of {args.model_type} ...")
    config = BertConfig.from_pretrained(args.model_checkpoint)
    # label
    config.id2label = args.id2label
    config.label2id = args.label2id
    config.num_labels = args.num_labels
    config.problem_type = args.problem_type

    # token label
    config.token_id2label = args.token_id2label
    config.token_label2id = args.token_label2id
    config.num_token_labels = args.num_token_labels

    config.classifier_dropout = args.classifier_dropout

    if args.f_train_data_count:
        with open(args.f_train_data_count) as f:
            data_count = json.load(f)
            train_data_count = list(data_count["train"].values())
            args.label_weights = torch.tensor(
                [
                    max(train_data_count) / item if item != 0 else 0
                    for item in train_data_count
                ],
                dtype=torch.float32,
            ).to(args.device)

    config.args = args
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = (
        MODEL_CLASSES[args.model_type]
        .from_pretrained(
            args.model_checkpoint,
            config=config,
        )
        .to(args.device)
    )
    # if args.torch_compile:
    #     model = torch.compile(model)

    # Training
    if args.do_train:
        train_dataset = MYDATA(args.train_file)
        dev_dataset = MYDATA(args.dev_file)
        train(args, train_dataset, dev_dataset, model, tokenizer)
    # Testing
    save_weights = [
        file for file in os.listdir(args.output_dir) if file.endswith(".bin")
    ]
    if args.do_test:
        test_dataset = MYDATA(args.test_file)
        test(args, test_dataset, model, tokenizer, save_weights)
    if args.do_predict:
        test_dataset = MYDATA(args.test_file)
        logger.info("***** Running predicting *****")
        for save_weight in save_weights:
            logger.info(f"loading weights from {save_weight}...")
            model.load_state_dict(
                torch.load(os.path.join(args.output_dir, save_weight))
            )
            report, df = predict(args, test_dataset, model)
            # 将DataFrame保存到csv文件中
            report.to_csv(
                "{}/{}_classification_report.csv".format(args.output_dir, save_weight),
                index=False,
            )
            df.to_csv(
                "{}/{}_model_results.csv".format(args.output_dir, save_weight),
                index=False,
            )
