import json
from typing import AnyStr, Optional

from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset


class MYDATA(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, "rt") as f:
            # rt模式下，python在读取文本时会自动把\r\n转换成\n
            for idx, line in enumerate(f):
                # print(line.strip())
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataLoader(
    dataset,
    tokenizer,
    batch_size=None,
    max_length=512,
    shuffle=False,
    num_labels=None,
    problem_type=None,
    use_r_drop=False,
    mode: Optional[AnyStr] = "Test",
):
    if problem_type == "multi_label_classification":
        mlb = MultiLabelBinarizer(classes=range(num_labels))

    def collote_fn(batch_samples):
        batch_sentence, batch_label = [], []
        for num, sample in enumerate(batch_samples):
            if use_r_drop and mode == "Train":
                batch_sentence.append(sample["sentence"])
            batch_sentence.append(sample["sentence"])
            if problem_type == "multi_label_classification":
                label = sample["label"]
                batch_label.append(label)
            else:
                label = int(sample["label"])
                batch_label.append(label)
        if problem_type == "multi_label_classification":
            batch_label = mlb.fit_transform(batch_label).astype("float32")
        batch_inputs = tokenizer(
            batch_sentence,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if mode == "Predict":
            return {
                "batch_inputs": batch_inputs,
                "labels": batch_label,
                "sentences": batch_sentence,
            }
        return {"batch_inputs": batch_inputs, "labels": batch_label}

    return DataLoader(
        dataset,
        batch_size=(batch_size if batch_size else batch_size),
        shuffle=shuffle,
        collate_fn=collote_fn,
    )


# 借助word_ids 实现标签映射
def process_function(
    batch_tokens,
    batch_token_labels,
    max_length,
    tokenizer,
    padding="max_length",
    truncation=True,
):
    tokenized_exmaples = tokenizer(
        batch_tokens,
        max_length=max_length,
        truncation=truncation,
        is_split_into_words=True,
        padding=padding,
        return_tensors="pt",
    )
    labels = []
    for i, label in enumerate(batch_token_labels):
        word_ids = tokenized_exmaples.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_id])
        labels.append(label_ids)
    batch_token_labels = labels
    return tokenized_exmaples, batch_token_labels


def get_multitask_sequence_and_token_dataLoader(
    dataset,
    tokenizer,
    batch_size=None,
    max_length=512,
    shuffle=False,
    num_labels=None,
    problem_type=None,
    use_r_drop=False,
    mode: Optional[AnyStr] = "Test",
):
    if problem_type == "multi_label_classification":
        mlb = MultiLabelBinarizer(classes=range(num_labels))

    def collote_fn(batch_samples):
        batch_sentence, batch_labels, batch_tokens, batch_token_labels = [], [], [], []
        for sample in batch_samples:
            batch_sentence.append(sample["sentence"])
            # label
            if problem_type == "multi_label_classification":
                label = sample["label"]
                batch_labels.append(label)
            else:
                label = int(sample["label"])
                batch_labels.append(label)
            # token
            batch_tokens.append(list(sample["sentence"]))
            batch_token_labels.append(list(sample["ner_tags"]))

        if problem_type == "multi_label_classification":
            batch_labels = mlb.fit_transform(batch_labels).astype("float32")
        batch_inputs, batch_token_labels = process_function(
            batch_tokens,
            batch_token_labels,
            max_length,
            tokenizer,
            padding="max_length",
            truncation=True,
        )
        if mode == "Predict":
            return {
                "batch_inputs": batch_inputs,
                "labels": batch_labels,
                "sentences": batch_sentence,
                "batch_tokens": batch_tokens,
                "token_labels": batch_token_labels,
            }
        return {
            "batch_inputs": batch_inputs,
            "labels": batch_labels,
            "batch_tokens": batch_tokens,
            "token_labels": batch_token_labels,
        }

    return DataLoader(
        dataset,
        batch_size=(batch_size if batch_size else batch_size),
        shuffle=shuffle,
        collate_fn=collote_fn,
    )


if __name__ == "__main__":
    # train_dataset = MYDATA("../data/0214/train.json")
    # print(type(train_dataset))
    # print(train_dataset[1])
    from pprint import pprint

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/models/chinese-macbert-base"
    )
    # train_dataloader = get_dataLoader(train_dataset, tokenizer, 32, False)
    # for batch_data in train_dataloader:
    #     print(batch_data)

    train_dataset = MYDATA(
        "/mnt/bn/fulei-v6-hl-nas-mlx/mlx/workspace/saas/imbot/data/d0823/train_d0823.json"
    )
    pprint(train_dataset[100])
    train_dataloader = get_multitask_sequence_and_token_dataLoader(
        train_dataset, tokenizer, 4, 16, False, 185, "single_label_classification"
    )
    for batch_data in train_dataloader:
        pprint(batch_data)
        break

    train_dataloader = get_dataLoader(
        train_dataset, tokenizer, 4, 16, False, 185, "single_label_classification"
    )
    for batch_data in train_dataloader:
        pprint(batch_data)
        break
