import argparse
import csv
import json
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

random.seed(42)


def process_ner_tag(sentence, ner_tag):
    """处理 ner_tag

    Args:
        sentence (str): 待处理的句子 "我想听周杰伦的歌"
        ner_tag (list[tuple[str, str]]): 标注的 ner_tag [("周杰伦", "名字")]
    Returns:
        sentence_ner_tag (list): 每个字的 ner_tag
    """
    sentence_ner_token = ["O"] * len(sentence)
    for entity, entity_type in ner_tag:
        entity_type = ner_categories[entity_type]
        if entity in sentence:
            start = sentence.index(entity)
            end = start + len(entity)
            sentence_ner_token[start] = "B-" + entity_type
            for i in range(start + 1, end):
                sentence_ner_token[i] = "I-" + entity_type
    sentence_ner_tag = [ner_label2id[i] for i in sentence_ner_token]
    return sentence_ner_token, sentence_ner_tag


def write_data(f_r, f_train, f_test, f_eval, f_data_count, f_data_len):
    # 保证 data_count 的 key 值和 label2id 的 key 值顺序一致
    data_count = {
        "total": {label: 0 for label in label2id},
        "train": {label: 0 for label in label2id},
        "eval": {label: 0 for label in label2id},
        "test": {label: 0 for label in label2id},
    }
    data_len = []
    random_scale = 0.9
    if f_test is not None:
        # 如果有测试集，将数据分为训练集、测试集、验证集
        random_scale = 0.8
        f_test_w = open(f_test, "w")
    with open(f_r, "r", encoding="utf-8-sig") as f, open(
        f_train, "w"
    ) as f_train_w, open(f_eval, "w") as f_eval_w:
        reader = list(csv.DictReader(f))
        for line in tqdm(reader):
            # print(line)
            sentence = line["sentence"].strip()
            data_len.append(len(sentence))
            label = line["label"].strip().split(",")

            # 如果是 ner 任务，需要处理 ner_tag
            if "ner" in args.problem_type:
                try:
                    # if sentence == "这个宝马x3曜夜套装最低多少钱":
                    #     print("sentence: ", sentence)
                    #     print("ner_tag: ", line["ner_tag"])
                    ner_tag = line["ner_tag"].strip().split("\n")
                    ner_tag = [
                        (i.split("：")[1], i.split("：")[0])
                        for i in ner_tag
                        if i.split("：")[1] not in "车系品牌燃料形式驱动方式"
                    ]
                    sentence_ner_token, sentence_ner_tag = process_ner_tag(
                        sentence, ner_tag
                    )
                except:
                    sentence_ner_token = ["O"] * len(sentence)
                    sentence_ner_tag = [ner_label2id[i] for i in sentence_ner_token]

            # 如果当前这个句话没有标签，就给它一个无意图标签
            if not label[0]:
                label = ["无意图标签"]
            try:
                for i in label:
                    data_count["total"][i] += 1
                labelid = [label2id[i.strip()] for i in label if i in label2id]
                # 如果是单标签分类任务，只取第一个标签
                if "single_label_classification" in args.problem_type:
                    labelid = labelid[0]
                # if "ner" in args.problem_type:

            except:
                print("line: ", line)
                print("label: ", label)
                print("sentence: ", sentence)
                continue

            if "ner" in args.problem_type:
                data = {
                    "sentence": sentence,
                    "label": labelid,
                    "ner_tags": sentence_ner_tag,
                }
            else:
                data = {
                    "sentence": sentence,
                    "label": labelid,
                }
            if random.random() < random_scale:
                f_train_w.write(json.dumps(data, ensure_ascii=False) + "\n")
                for i in label:
                    data_count["train"][i] += 1
            else:
                if f_test is not None:
                    if random.random() < 0.5:
                        f_test_w.write(json.dumps(data, ensure_ascii=False) + "\n")
                        for i in label:
                            data_count["test"][i] += 1
                    else:
                        f_eval_w.write(json.dumps(data, ensure_ascii=False) + "\n")
                        for i in label:
                            data_count["eval"][i] += 1
                else:
                    f_eval_w.write(json.dumps(data, ensure_ascii=False) + "\n")
                    for i in label:
                        data_count["eval"][i] += 1

    if f_data_count is not None:
        data_count = pd.DataFrame(data_count)
        data_count.to_csv(f_data_count, encoding="utf-8-sig", index=True)
        f_data_count_json = f_data_count.replace(".csv", ".json")
        data_count.T.to_json(
            f_data_count_json, orient="index", force_ascii=False, indent=4
        )

    if f_data_len is not None:
        data_len_dict = {}
        data_len_dict["avg_len"] = np.mean(data_len)
        data_len_dict["max_len"] = max(data_len)
        data_len_dict["min_len"] = min(data_len)
        data_len_dict["median_len"] = np.median(data_len)

        data_len_dict = pd.DataFrame(data_len_dict, index=["count"])
        # 只取 avg_len, max_len, min_len
        data_len_dict = data_len_dict[["avg_len", "max_len", "min_len", "median_len"]]
        data_len_dict.to_csv(f_data_len, encoding="utf-8-sig", index=True)


def get_label_config(f_label, f_config):
    with open(f_label, "r", encoding="utf-8") as f, open(f_config, "w") as f_config_w:
        reader = csv.DictReader(f)
        label2id = {line["label"].strip(): num for num, line in enumerate(reader)}
        json.dump(label2id, f_config_w, ensure_ascii=False, indent=4)
    return label2id


def get_token_label_config(f_ner_dict, f_ner_config):
    with open(f_ner_dict) as f_ner_dict, open(f_ner_config, "w") as f_ner_config_w:
        ner_categories = json.load(f_ner_dict)
        token_id2label = {0: "O"}
        for label in ner_categories.values():
            token_id2label[len(token_id2label)] = "B-" + label
            token_id2label[len(token_id2label)] = "I-" + label
        token_label2id = {v: k for k, v in token_id2label.items()}
        json.dump(token_label2id, f_ner_config_w, ensure_ascii=False, indent=4)
        return ner_categories, token_label2id


if __name__ == "__main__":
    # sentence_ner_tag = process_ner_tag(
    #     sentence="你稍等一下我先不跟你说我现在付个钱，然后一会我给你打过去好吗？",
    #     ner_tag=[("现在", "time"), ("一会", "time"), ("稍等一下", "time")],
    # )
    # print(sentence_ner_tag)

    parser = argparse.ArgumentParser()
    parser.add_argument("--f_r", type=str, default=None)
    parser.add_argument(
        "--problem_type",
        choices=["single_label_classification", "multi_label_classification", "ner"],
        type=str,
        default=None,
    )
    parser.add_argument("--f_train", type=str, default=None)
    parser.add_argument("--f_test", type=str, default=None)
    parser.add_argument("--f_eval", type=str, default=None)
    parser.add_argument("--f_label", type=str, default=None, required=False)
    parser.add_argument("--f_config", type=str, default=None, required=True)
    parser.add_argument("--f_ner_dict", type=str, default=None, required=False)
    parser.add_argument("--f_ner_config", type=str, default=None, required=True)
    parser.add_argument("--f_data_count", type=str, default=None, required=False)
    parser.add_argument("--f_data_len", type=str, default=None, required=False)
    args = parser.parse_args()

    if args.f_label is not None:
        print("get label2id from label file")
        label2id = get_label_config(args.f_label, args.f_config)
    else:
        print("get label2id from config file")
        with open(args.f_config, "r") as f:
            label2id = json.load(f)

    print("get ner_label2id from ner_label file")
    ner_categories, ner_label2id = get_token_label_config(
        args.f_ner_dict, args.f_ner_config
    )

    print("split and write data to file")
    write_data(
        args.f_r,
        args.f_train,
        args.f_test,
        args.f_eval,
        args.f_data_count,
        args.f_data_len,
    )
