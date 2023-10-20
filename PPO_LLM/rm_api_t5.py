import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, T5Model

# CUDA 0




# Import our models. The package will take care of downloading the models automatically
# sim_tokenizer = AutoTokenizer.from_pretrained("/root/hdd/sup-simcse-roberta-large")
sim_tokenizer = AutoTokenizer.from_pretrained("/root/hdd/gense-base-plus")
# sim_tokenizer = AutoTokenizer.from_pretrained("/root/hdd/data/my-sup-simcse-test")
sim_model = T5Model.from_pretrained(
    # "/root/hdd/sup-simcse-roberta-large", torch_dtype=torch.float16
    # "/root/hdd/data/my-sup-simcse-test", torch_dtype=torch.float16
    "/root/hdd/gense-base-plus", torch_dtype=torch.float16
).cuda().half()
dis_model = AutoModelForSequenceClassification.from_pretrained(
    "/root/hdd/nli-deberta-v3-large", torch_dtype=torch.float16
).cuda()
dis_tokenizer = AutoTokenizer.from_pretrained("/root/hdd/nli-deberta-v3-large")

import torch
import torch.nn.functional as F

similarity_func = lambda s1, s2: torch.nan_to_num(
    F.cosine_similarity(torch.nan_to_num(s1), torch.nan_to_num(s2), dim=-1)
)

def cal_cos_sim(query, response):
    texts = query.copy()
    texts.extend(response)

    texts = [s + ' Question: what can we draw from the above sentence?' for s in texts]
    # print(texts)


    input_features = sim_tokenizer(texts, add_special_tokens=True, padding=True, return_tensors='pt').to('cuda')
    # decoder_start_token_id = sim_model._get_decoder_start_token_id()
    decoder_start_token_id = 0
    input_features['decoder_input_ids'] = torch.full([input_features['input_ids'].shape[0], 1], decoder_start_token_id).to('cuda')

    # inference
    with torch.no_grad():
        outputs = sim_model(**input_features, output_hidden_states=True, return_dict=True)
        # print(len(outputs))
        last_hidden = outputs.last_hidden_state
        embeddings = last_hidden[:, 0].float().cpu()
    # print(sent_embs)
    cosine_scores = []
    for i in range(len(query)):
        cosine_score = similarity_func(embeddings[i], embeddings[i+len(query)])  # same as 1-cosine
        cosine_scores.append(cosine_score)
        # cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

    # return cosine_scores
    query_embeddings = embeddings[: len(query)]
    response_embeddings = embeddings[len(query) :]

    cosine_scores = similarity_func(query_embeddings, response_embeddings)

    return cosine_scores.tolist()


def dis_check(query, response):
    features = dis_tokenizer(
        query, response, padding=True, truncation=True, return_tensors="pt"
    ).to("cuda")

    dis_model.eval()
    with torch.no_grad():
        scores = dis_model(**features).logits
    label_mapping = ["contradiction", "entailment", "neutral"] 
    id2label = ({"0": "contradiction", "1": "entailment", "2": "neutral"},)
    # labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
    probs = F.softmax(scores, dim=1)
    max_indices = torch.argmax(probs, dim=1)
    labels = [label_mapping[score_max] for score_max in max_indices]
    confidences = torch.max(probs, dim=1).values
    # print(labels,confidences)

    return labels, scores.tolist()



# target hard positive 0.85
target_positive = 0.92
# target hard negative 0.6?
# target_negative = 0.6
target_negative = 0.60

def cos2score(sim_score, labels):
    score1 = []
    for i in range(len(sim_score)):
        if labels[i] == "entailment":
            
            tempScore = target_positive - sim_score[i]
            if sim_score[i] < 0.7: #clip
                tempScore = -tempScore
            score1.append(
                tempScore * 15
            )  # lower than target_positive is good
        elif labels[i] == "contradiction":
            tempScore = (sim_score[i] - target_negative)
            if tempScore > 0:
                tempScore *= 1.2
            score1.append(
                tempScore * 20
            )  # higher than target_negative is good

    return score1

# A basketball team of 8 boys is doing a hand huddle
# A basketball team of 8 girls is doing a hand huddle.

def label2score(raw_label, pred_label, pred_score):
    id2label = {"contradiction": 0, "entailment": 1, "neutral": 2}
    score2 = []

    for i in range(len(raw_label)):
        if str(raw_label[i]) == str(pred_label[i]):
            probs = F.softmax(torch.Tensor(pred_score[i]), dim=-1)
            confidences = torch.max(probs, dim=-1).values.item()
            score2.append((confidences - 0.93) * 30)
            # score2.append(1.5)
            # score2.append(pred_score[i][id2label[raw_label[i]]] / 2)
        elif raw_label[i] == "contradiction" and pred_label[i] == "neutral":
            score2.append(-1)  
        # elif raw_label[i] == "entailment" and pred_label[i] == "neutral":
        #     score2.append(pred_score[i][id2label["entailment"]])
        else:
            # if pred_score[i][id2label[raw_label[i]]] < 0:
            # score2.append(pred_score[i][id2label[raw_label[i]]] * 2)
            score2.append(pred_score[i][id2label[raw_label[i]]] * 1.5)
            # else:
            #     score2.append(-4)
    return score2


import re


def contains_chinese_characters(text):
    # pattern = re.compile(r"[\u4e00-\u9fff]")  
    # result = re.search(pattern, text)
    # return result is not None
    pattern = re.compile(r"[^\x00-\x7F]")
    result = re.search(pattern, text)
    return result is not None


def complete_score(response, pre_result, raw):
    # pre_result = []
    # print(len(response), len(pre_result))
    for ind, item in enumerate(response):
        if len(item) < 5:
            pre_result[ind] = -7
            continue
        if contains_chinese_characters(item):
            if pre_result[ind] < 0:
                pre_result[ind] -= 7.5
            pre_result[ind] = -8
            continue

        length_smaller = len(raw[ind].split(" ")) - len(item.split(" "))

        if item[1:-3] in raw[ind]:
            # print(item)
            pre_result[ind] = -5
        elif len(raw[ind].split(" ")) - len(item.split(" ")) > 5:
            #len("Mother and Child, seems like it is cold there.".split(" ")) - len('Person."'.split(" "))
            pre_result[ind] -= 4

        if length_smaller > 0 :
            pre_result[ind] -= length_smaller / 3

        # check_item = item
        # if 'Original Sentence:' in item:
        #     check_item = check_item.split('Original Sentence:')[0].strip()
        # print(result)
        # if item.index('Original Sentence:') > 20:
        #     score_c.append(-1)
        # else:
        #     score_c.append(-2)
        if item[-1] != "." or item[-1] != '"':  # or item[-1] != ')'
            # if len(item.split(' ')) <= 10:
            #     score_c.append(-2.5)
            # el
            # if '(' in item and item.index('(') > 20:
            #     score_c.append(0)
            # else:
            pre_result[ind] -= 3.5
        else:
            pre_result[ind] += 1
    return pre_result


import json


def append_to_jsonl(data, file_path):
    # data['score'] = res.copy()
    with open(file_path, "a") as file:
        # for key, values in data.items():
        #     json_obj = {'key': key, 'values': values}
        json_line = json.dumps(data, ensure_ascii=False)
        file.write(json_line + "\n")


from fastapi import FastAPI
from typing import List, Dict

app = FastAPI()


# {'query': ['test', 'test2'], 'response': ['test', 'test2']}
def preprocess(querys):
    pro_querys = []
    score_p = []
    for item in querys:
        check_item = item

        if "\n" in item:
            if contains_chinese_characters(item):
                score_p.append(-5)
            else:
                score_p.append(-1.5)
            check_item = check_item.split("\n")[0].strip()

        elif 'Original Sentence:' in item:
            check_item = check_item.split('Original Sentence:')[0].strip()
            score_p.append(-1)
        elif 'Note:' in item:
            check_item = check_item.split('Note:')[0].strip()
            score_p.append(-1)
        elif 'Variation:' in item:
            check_item = check_item.split('Variation:')[0].strip()
            score_p.append(-1)
        else:
            score_p.append(0.5)

        pro_querys.append(check_item)

    return pro_querys, score_p


# bop
@app.post("/reward")
def get_reward_score(inputs: Dict[str, List[str]]) -> List[float]:
    response, score_p = preprocess(inputs["response"])
    # print(response)
    inputs["response"] = response
    score_sim = cal_cos_sim(inputs["query"], response)
    # print(score_sim)
    score1 = cos2score(score_sim, inputs["label"])
    pred_labels, pred_scores = dis_check(inputs["query"], response)
    score2 = label2score(inputs["label"], pred_labels, pred_scores)
    pre_result = [x + y + z for x, y, z in zip(score1, score2, score_p)]

    result = complete_score(response, pre_result, inputs["query"])
    # result = [-10 if x < -10 else x for x in result]

    ttt = [
        [x, y, z, a]
        for x, y, z, a in zip(inputs["query"], response, result, inputs["label"])
    ]
    # print(ttt)

    append_to_jsonl(ttt, "/root/hdd/data/log1.jsonl")

    return result


# if __name__ == "__main__":
import uvicorn

uvicorn.run(app, host="127.0.0.1", port=12345)
