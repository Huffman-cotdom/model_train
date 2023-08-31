import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import os, json, csv

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# config_path = 'customer/data/replace_data/customer_replace_label2id.json'
# with open(os.path.join(dir_path, config_path), 'r') as f:
#     label2id = json.load(f)
#     id2label = {v: k for k, v in label2id.items()}

# model_checkpoint = 'customer/model_res/model'
# model_checkpoint = os.path.join(dir_path, model_checkpoint)
# num_labels = 8
# problem_type = 'multi_label_classification'

# tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
# model = BertForSequenceClassification.from_pretrained(
#     model_checkpoint,
#     num_labels=num_labels,
#     problem_type=problem_type,
# ).to('cuda')


def test_loop(sentence):
    inputs = tokenizer(
        sentence,
        max_length=128,
        truncation=True,
        return_tensors="np",
    )
    model.eval()
    inputs = {k: torch.tensor(v).to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        probs = (probs > 0.5).int().cpu().numpy()
        pred = [id2label[num] for num, i in enumerate(probs[0]) if i == 1]
        return pred


# data_path = 'get_quanyong_data/replace_data.csv'
# data_path = os.path.join(dir_path, data_path)
# customer_model_res_path = 'get_quanyong_data/customer_model_res.csv'
# customer_model_res_path = os.path.join(dir_path, customer_model_res_path)
# with open(data_path) as f, open(customer_model_res_path, 'w') as f_res:
#     w = csv.writer(f_res)
#     r = csv.reader(f)
#     next(r)
#     for line in tqdm(r, total=65552):
#         if line == []:
#             continue
#         id, role, sentence = line
#         if role == '客户':
#             pred = test_loop(sentence)
#             w.writerow([id, role, sentence, pred])
#         else:
#             w.writerow([id, role, sentence])

config_path = 'seller/data/replace_data/seller_replace_label2id.json'
with open(os.path.join(dir_path, config_path), 'r') as f:
    label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

model_checkpoint = 'seller/model_res/model'
model_checkpoint = os.path.join(dir_path, model_checkpoint)
num_labels = 5
problem_type = 'multi_label_classification'

tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
model = BertForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    problem_type=problem_type,
).to('cuda')


data_path = 'get_quanyong_data/customer_model_res.csv'
data_path = os.path.join(dir_path, data_path)
total_model_res_path = 'get_quanyong_data/total_model_res.csv'
total_model_res_path = os.path.join(dir_path, total_model_res_path)
with open(data_path) as f, open(total_model_res_path, 'w') as f_res:
    w = csv.writer(f_res)
    w.writerow(['id', 'role', 'sentence', 'label'])
    r = csv.reader(f)
    for line in tqdm(r, total=65552):
        if line == []:
            continue
        role = line[1]
        if role == '销售':
            id, role, sentence = line
            pred = test_loop(sentence)
            w.writerow([id, role, sentence, pred])
        else:
            w.writerow(line)
