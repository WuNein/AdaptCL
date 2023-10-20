import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


dis_model = AutoModelForSequenceClassification.from_pretrained('/root/hdd/nli-deberta-v3-large', device_map={'':0}, torch_dtype=torch.bfloat16)
dis_tokenizer = AutoTokenizer.from_pretrained('/root/hdd/nli-deberta-v3-large')


df = pd.read_csv('/root/data/ppo4cl/pro_instruct_2w_stage2.csv')


correct_predictions = 0
total_predictions = 0

#anchor,pos,neg

# sent0,sent1,hard_neg

batch_size = 128
for i in range(0, len(df), batch_size):
    batch = df[i:i+batch_size]
    anchors = batch['anchor'].tolist()
    positives = batch['pos'].tolist()
    negatives = batch['neg'].tolist()

    features_positive = dis_tokenizer(anchors, positives, padding=True, truncation=True, return_tensors="pt")
    features_negative = dis_tokenizer(anchors, negatives, padding=True, truncation=True, return_tensors="pt")
    dis_model.eval()

    with torch.no_grad():
        scores_positive = dis_model(**features_positive).logits
        # scores_negative = dis_model(**features_negative).logits

        label_mapping = ['contradiction', 'entailment', 'neutral']

        max_indices_positive = torch.argmax(scores_positive, dim=1)
        # max_indices_negative = torch.argmax(scores_negative, dim=1)

        labels_positive = [label_mapping[score_max] for score_max in max_indices_positive]
        # labels_negative = [label_mapping[score_max] for score_max in max_indices_negative]

        # valid
        # correct_predictions += sum(label_p == 'entailment' and label_n != 'entailment' for label_p, label_n in zip(labels_positive, labels_negative))
        correct_predictions += sum(label_p == 'entailment' for label_p in labels_positive)
        # correct_predictions += sum(label_n != 'entailment' for label_n in labels_negative)

        total_predictions += len(batch)

print(f'Correct Predictions: {correct_predictions}/{total_predictions}')
