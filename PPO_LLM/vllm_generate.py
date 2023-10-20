from vllm import LLM, SamplingParams
from time import time
from datasets import load_dataset, load_from_disk, concatenate_datasets
import json

# /root/hdd/data/runs_wi/merged_stage2_final

llm = LLM(
    model="/root/hdd/data/runs_wi/merged_t5_final",
    tokenizer="/root/hdd/llm/WizardLM-7B-v1",
    dtype="bfloat16",
)
sampling_params = SamplingParams(temperature=0.8, top_p=1.0, max_tokens=48, best_of=3)

train_dataset_snli = load_from_disk("/root/hdd/data/snli_small").shuffle(seed=42)
train_dataset_wiki = load_from_disk("/root/hdd/data/wiki_small").shuffle(seed=42)#.select(range(40000))

train_dataset = concatenate_datasets([train_dataset_snli, train_dataset_wiki])

train_dataset = train_dataset.filter(lambda example: example["label"] == "entailment")

test_dataset = train_dataset.shuffle(seed=721).select(range(40000))
print(test_dataset)

pos_arr = []
neg_arr = []

for indx, item in enumerate(test_dataset):
    question = item["premise"]
    review_pos = f"""Generate a hard positive variation of Original Sentence while ensuring it retains the same meaning, does not alter the original sentiment, exhibits different syntactical and grammatical structure, and do not introduce and add any additional information. Only one complete response sentence in English is need.
Original Sentence: "In County Limerick, Ireland, Oola Castle is a tower house that can be found."
Variation: "In the County of Limerick, Ireland, stands Oola Castle, a tower house that is to be found."
Original Sentence: "He rarely missed a game over the following three seasons." 
Variation: "Over the following three seasons, he rarely missed a game."
Original Sentence: "The first edition was published in February 1807 in Vienna."
Variation: "It was in Vienna that the first edition saw its publication in February 1807."
Original Sentence: "{question}" 
"""
    query_pos = f"""{review_pos}\n\n### Response: Variation: """  # wizard 7b
    pos_arr.append(query_pos)

    review_neg = f"""Please provide a hard negative variation of Original Sentence, ensuring it has a completely different meaning, similar syntax and grammar, and does not contain excessive additional information. Only one complete response sentence in English is need.
Original Sentence: "A father and a son are tucked tightly together while sleeping."
Variation: "A father and a son are tucked loosely together while sleeping."
Original Sentence: "He found the book fascinating." 
Variation: "The book failed to captivate him despite its initial promise."
Original Sentence:  "Her machinery and hull would also be repaired and renewed."
Variation: "Neither her machinery nor hull would receive any repairs or updates."
Original Sentence: "He found the boy bites the dog." 
Variation: "He found the dog bites the boy."
Original Sentence: "{question}"
"""
    query_neg = f"""{review_neg}\n\n### Response: Variation: """  # wizard 7b
    neg_arr.append(query_neg)

    if indx > 0 and indx % 500 == 0:
        # outputs = llm.generate(neg_arr, sampling_params)
        # matched_neg = []
        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     input_text = output.prompt.split("Original Sentence: ")[-1].split("\n")[0].strip()[1:-1]
        #     matched_neg.append(
        #         {
        #             "anchor": input_text,
        #             "hard": generated_text.strip(),
        #             "label": "contradiction",
        #         }
        #     )
        # jsonl_data_neg = "\n".join(json.dumps(data, ensure_ascii=False) for data in matched_neg)

        # with open("test_gen_neg.jsonl", "a") as f:
        #     f.write(jsonl_data_neg + '\n')


        outputs = llm.generate(pos_arr, sampling_params)
        matched_pos = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            input_text = output.prompt.split("Original Sentence: ")[-1].split("\n")[0].strip()[1:-1]
            matched_pos.append(
                {
                    "anchor": input_text,
                    "hard": generated_text.strip(),
                    "label": "entailment",
                }
            )
        jsonl_data_pos = "\n".join(json.dumps(data, ensure_ascii=False) for data in matched_pos)
        
        with open("test_gen_pos_t5_4w.jsonl", mode="a") as f:
            f.write(jsonl_data_pos + '\n')

        pos_arr = []
        neg_arr = []
