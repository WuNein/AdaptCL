import json
import math
import time
import requests
#from llama_attn_hijack_xformers import hijack_llama_attention

#hijack_llama_attention()

# OSError: Token is required (`token=True`), but no token found. You need to provide a token or be logged in to Hugging Face with `huggingface-cli login` or `huggingface_hub.login`. See https://huggingface.co/settings/tokens.

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk, concatenate_datasets
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, LlamaTokenizer, HfArgumentParser, pipeline, AutoTokenizer, AutoModel
import transformers
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.trainer.ppo_trainer import PPODecorators, logprobs_from_logits
import trl
import os
from transformers.optimization import get_scheduler
# tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    model_name: Optional[str] = field(
        default="WizardLM/WizardLM-7B-V1.0", metadata={"help": "the model name"}
    )
    tokenizer_name: Optional[str] = field(
        default="WizardLM/WizardLM-7B-V1.0", metadata={"help": "the tokenizer name"}
    )
    # reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(
        default='tensorboard', metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    output_max_length: Optional[int] = field(
        default=48, metadata={"help": "maximum length for generation"}
    )
    mini_batch_size: Optional[int] = field(
        default=8, metadata={"help": "the PPO minibatch size"}
    )
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(
        default=4, metadata={"help": "the number of ppo epochs"}
    )
    # gradient_accumulation_steps: Optional[int] = field(
    #     default=4, metadata={"help": "the number of gradient accumulation steps"}
    # )
    adafactor: Optional[bool] = field(
        default=False, metadata={"help": "whether to use the adafactor optimizer"}
    )
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "whether to early stop"}
    )
    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "kl target for early stopping"}
    )
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(
        default=False, metadata={"help": "whether to use the batched text gen"}
    )
    save_freq: Optional[int] = field(
        default=200, metadata={"help": "n steps to save the model"}
    )
    output_dir: Optional[str] = field(
        default="/root/hdd/data/runs_wi/", metadata={"help": "n steps to save the model"}
    )
    # seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=100000, metadata={"help": "number of it"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Initial KL penalty coefficient (used for adaptive and linear control)"
        },
    )

    adap_kl_ctrl: Optional[bool] = field(
        default=True, metadata={"help": "Use adaptive KL control, otherwise linear"}
    )

    lr_scheduler_type : Optional[str] = field(
        default="cosine", metadata={"help": "lr_scheduler_type"}
    )


    local_rank: Optional[int] = field(
        default=0, metadata={"help": "nvidia ddp"}
    )


parser = HfArgumentParser(ScriptArguments)
# script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
script_args, unknown_args = parser.parse_known_args()

gradient_accumulation_steps = script_args.batch_size // script_args.mini_batch_size

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // world_size

print('gradient_accumulation_steps', gradient_accumulation_steps)
# reward_model_name = script_args.reward_model_name
# dataset_name = "lvwerra/stack-exchange-paired"
# FutureWarning: `logging_dir` is deprecated and will be removed in version 0.18.0 of 🤗 Accelerate. Use `project_dir` instead.
config = PPOConfig(
    accelerator_kwargs={"project_dir": '/root/data/ppo4cl/logs'},
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
    remove_unused_columns=False,
)

print(config)

# SNLI
# SNLI dataset

train_dataset_snli = load_from_disk("/root/hdd/data/snli_small").shuffle(seed=42)
train_dataset_wiki = load_from_disk("/root/hdd/data/wiki_small").shuffle(seed=42) #.select(range(40000))

train_dataset = concatenate_datasets([train_dataset_snli, train_dataset_wiki])

# dataset = dataset.filter(lambda example: example['label'] != 'neutral')

# train_dataset = train_dataset.shuffle(seed=42).select(range(40000))
train_dataset = train_dataset.shuffle(seed=114514).select(range(20000))
print(train_dataset)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
# sent_kwargs = {
#     "return_all_scores": True,
#     "function_to_apply": "none",
#     "batch_size": 16,
#     "truncation": True,
# }

tokenizer = LlamaTokenizer.from_pretrained(script_args.tokenizer_name)

# tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, trust_remote_code=True)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

# if getattr(tokenizer, "pad_token", None) is None:
#     tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"
# tokenizer.pad_token_id = 0  # unk
# tokenizer.bos_token_id = 1
# tokenizer.eos_token_id = 2

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
# def build_dataset(
#     tokenizer,
#     # dataset_name="snli",
# ):
"""
Build dataset for training. This builds the dataset from `load_dataset`, one should
customize this function to train the model on its own dataset.

Args:
    dataset_name (`str`):
        The name of the dataset to be loaded.

Returns:
    dataloader (`torch.utils.data.DataLoader`):
        The dataloader for the dataset.
"""

# load imdb with datasets
# ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
# ds = load_from_disk(dataset_name, split="train")
# original_columns = ds.column_names


def preprocess_function(examples):
    # examples {'premise': []

    new_examples = {
        "raw": [],
        "label": [],
        "query": [],
        "input_ids": [],
        # "attention_mask": [],
    }
    for ind in range(len(examples["premise"])):
        # query = "Question: " + question + "\n\nAnswer: "
        # positive and negatives
        # label_map = {
        #     0: "entailment",
        #     1: "neutral",
        #     2: "contradiction"
        # }
        # different from nli-deberta-v3-large
        question = examples["premise"][ind]
        review = ""
        label = ""
        if examples["label"][ind] == "entailment" or examples["label"][ind] == 0:
            review = f"""Generate a hard positive variation of Original Sentence while ensuring it retains the same meaning, does not alter the original sentiment, exhibits different syntactical and grammatical structure, and do not introduce and add any additional information. Only one complete response sentence in English is need.
Original Sentence: "In County Limerick, Ireland, Oola Castle is a tower house that can be found."
Variation: "In the County of Limerick, Ireland, stands Oola Castle, a tower house that is to be found."
Original Sentence: "He rarely missed a game over the following three seasons." 
Variation: "Over the following three seasons, he rarely missed a game."
Original Sentence: "The first edition was published in February 1807 in Vienna."
Variation: "It was in Vienna that the first edition saw its publication in February 1807."
Original Sentence: "{question}" 
"""
            label = "entailment"
        elif examples["label"][ind] == "contradiction" or examples["label"][ind] == 2:
            review = f"""Please provide a hard negative variation of Original Sentence, ensuring it has a completely different meaning, similar syntax and grammar, and does not contain excessive additional information. Only one complete response sentence in English is need.
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
            label = "contradiction"
        else:
            continue
        # query = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n###Human: {review} \n###Assistant: Variation: " #vicuna 7b
        query = f"""{review}\n\n### Response: Variation: """ # wizard 7b
        # query = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: {}ASSISTANT: Variation: ".format(review) # vi 7b 1.3 
        tokenized_question = tokenizer(query, truncation=True, max_length=512)
        new_examples["raw"].append(question)
        new_examples["label"].append(label)
        new_examples["query"].append(query)
        new_examples["input_ids"].append(tokenized_question["input_ids"])
        # new_examples["attention_mask"].append(tokenized_question["attention_mask"])

    return new_examples

    # ds = train_dataset.map(
    #     preprocess_function,
    #     batched=True,
    #     num_proc=num_proc,
    #     # remove_columns=original_columns,
    # )
    # # ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    # ds.set_format(type="torch")
    # return ds


num_proc = 16


# We retrieve the dataloader by calling the `build_dataset` function.
# dataset = build_dataset(train_dataset, tokenizer)
dataset = train_dataset.map(
    preprocess_function, batched=True, batch_size=128, num_proc=num_proc
)

dataset.set_format(type="torch")
# print(dataset[:10])


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])



# CustomDataCollator(tokenizer)

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"], #llama
    # target_modules=["query_key_value"], #chatglm2
)

        # model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
# model = AutoModel.from_pretrained(config.model_name, device_map=device_map, torch_dtype=torch.bfloat16)
# # model.lm_head = model.transformer.output_layer
# model = AutoModelForCausalLMWithValueHead.from_pretrained(model,peft_config=lora_config)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    # model,
    config.model_name,
    # config.model_name,
    # device_map={"": current_device},
    device_map=device_map,
    peft_config=lora_config,
    torch_dtype=torch.bfloat16,
)

# assert load_valuehead_params(model, model_args.reward_model), "Reward model is not correctly loaded."
# def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
#     valuehead_file = os.path.join(checkpoint_dir, VALUE_HEAD_FILE_NAME)
#     if not os.path.exists(valuehead_file):
#         logger.warning("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
#         return False
#     valuehead_state_dict = torch.load(valuehead_file, map_location="cpu")
#     model.register_buffer("reward_head_weight", valuehead_state_dict["summary.weight"])
#     model.register_buffer("reward_head_bias", valuehead_state_dict["summary.bias"])
#     model.register_buffer("default_head_weight", torch.zeros_like(valuehead_state_dict["summary.weight"]))
#     model.register_buffer("default_head_bias", torch.zeros_like(valuehead_state_dict["summary.bias"]))
#     return True

# model = AutoModelForCausalLMWithValueHead.from_pretrained(
#     # model,
#     config.model_name,
#     # config.model_name,
#     # device_map={"": current_device},
#     device_map=device_map,
#     peft_config=lora_config,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# )
# model = AutoModel.from_pretrained(script_args.model_name, trust_remote_code=True, device_map={"":0}, torch_dtype=torch.bfloat16)


# from lion_pytorch import Lion
# optimizer = Lion(filter(lambda p: p.requires_grad, model.parameters()), lr=script_args.learning_rate / 3)
# optimizer = Lion(filter(lambda p: p.requires_grad, model.parameters()), lr=script_args.learning_rate / 3)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=script_args.learning_rate)
# optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
# if self.dataset is not None:
#             self.dataloader = self.prepare_dataloader(self.dataset, data_collator)

lr_scheduler = get_scheduler(
    script_args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=20,
    num_training_steps=(script_args.ppo_epochs * math.ceil(len(dataset) / script_args.batch_size))
)

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    # data_collator=CustomDataCollator(tokenizer),
    optimizer=optimizer,
    lr_scheduler=lr_scheduler
)

ppo_trainer._signature_columns = ["label", "query", "response", "raw"]
# ppo_trainer._set_signature_columns_if_needed()

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
# device = ppo_trainer.accelerator.device
# if ppo_trainer.accelerator.num_processes == 1:
#     device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
# sentiment_pipe = pipeline(
#     "sentiment-analysis",
#     model=reward_model_name,
#     device_map={"": current_device},
#     model_kwargs={"load_in_8bit": True},
#     tokenizer=tokenizer,
#     return_token_type_ids=False,
# )

# 会在另外一张卡上跑，是MoE的RM

# Avoid runtime error in model.generate(do_sample=True).
# class InvalidScoreLogitsProcessor(LogitsProcessor):

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         if torch.isnan(scores).any() or torch.isinf(scores).any():
#             scores.zero_()
#             scores[..., 0] = 1.0
#         return scores


# def get_logits_processor() -> LogitsProcessorList:
#     logits_processor = LogitsProcessorList()
#     logits_processor.append(InvalidScoreLogitsProcessor())
#     return logits_processor

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
# /root/hdd/llm/vicuna-7b/generation_config.json
#   "bos_token_id": 0,
#   "eos_token_id": 1,
#   "pad_token_id": 0,
# generation_kwargs = {
#     "min_length": 10,
#     "top_k": 0.0,
#     # "top_k": 0,
#     # "top_p": 1.0, #1.0 不行
#     "top_p": 0.80,
#     "do_sample": True,
#     "pad_token_id": 0,
#     "eos_token_id": 2,
#     # "eos_token_id": -1,
# }

# class InvalidScoreLogitsProcessor(LogitsProcessor):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         if torch.isnan(scores).any() or torch.isinf(scores).any():
#             scores.zero_()
#             scores[..., 5] = 5e4
#         return scores

# logits_processor = None
# if logits_processor is None:
#     logits_processor = LogitsProcessorList()
# logits_processor.append(InvalidScoreLogitsProcessor())

# Avoid runtime error in model.generate(do_sample=True).
class InvalidScoreLogitsProcessor(LogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 0] = 1.0
        return scores


def get_logits_processor() -> LogitsProcessorList:
    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    return logits_processor

logits_processor = get_logits_processor()

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

generation_kwargs = {
    "min_length": 10,
    'max_new_tokens': script_args.output_max_length,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": 0,
    "eos_token_id": 2,
    # 'num_beams': 1,
    # 'temperature': 0.8, #?
    # 'length_penalty' : 1.2, #?
    "logits_processor": logits_processor
}
#https://github.com/lvwerra/trl/issues/203
output_min_length = 10
output_max_length = script_args.output_max_length
# output_length_sampler = LengthSampler(output_min_length, output_max_length)

url = "http://127.0.0.1:12345/reward"
max_retries = 3
retry_interval = 10

# @PPODecorators.empty_cuda_cache()
# def batched_forward_pass(
#     self,
#     model: AutoModelForCausalLMWithValueHead,
#     queries: torch.Tensor,
#     responses: torch.Tensor,
#     model_inputs: dict,
#     return_logits: bool = False
# ):
#     r"""
#     Calculates model outputs in multiple batches.

#     Subclass and override to inject custom behavior.
#     """
#     bs = len(queries)
#     fbs = self.config.mini_batch_size
#     all_logprobs = []
#     all_logits = []
#     all_masks = []
#     all_values = []

#     for i in range(int(bs / fbs)):
#         input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
#         query_batch = queries[i * fbs : (i + 1) * fbs]
#         response_batch = responses[i * fbs : (i + 1) * fbs]
#         input_ids = input_kwargs["input_ids"] # left-padded sequences

#         if self.is_distributed: # re-generate them to adapt padded inputs
#             input_kwargs["attention_mask"] = self.data_collator.get_attention_masks(input_ids, device=self.current_device)
#             input_kwargs["position_ids"] = self.data_collator.get_position_ids(input_ids, device=self.current_device)

#         logits, _, values = model(**input_kwargs)
#         logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

#         values = values.transpose(0, 1)
#         masks = torch.zeros_like(input_ids)

#         for j in range(fbs):
#             start = len(query_batch[j]) - 1
#             start += (input_ids[j] != self.tokenizer.pad_token_id).nonzero()[0].item()
#             end = start + len(response_batch[j])
#             masks[j][start:end] = 1
#             if len(masks[j][start:end]) < 2:
#                 raise ValueError("Responses are too short. Make sure they are at least 4 tokens long.")

#         if return_logits:
#             all_logits.append(logits)
#         else:
#             del logits
#         all_values.append(values)
#         all_logprobs.append(logprobs)
#         all_masks.append(masks)

#     return (
#         torch.cat(all_logprobs),
#         torch.cat(all_logits)[:, :-1] if return_logits else None,
#         torch.cat(all_values)[:, :-1],
#         torch.cat(all_masks)[:, :-1],
#     )


# print(config.total_ppo_epochs)
# print('dataloader length', len(ppo_trainer.dataloader) )

# one epoch
for step_count, batch in tqdm(enumerate(ppo_trainer.dataloader), total = len(ppo_trainer.dataloader)):
    # print(iter_counter)

    # if epoch >= config.total_ppo_epochs: # max epoch
    #     print('max-epoch')
    #     break

    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=None,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(
        response_tensors, skip_special_tokens=True
    )

    # new_examples = {
    #     "raw": [],
    #     "label": [],
    #     "query": [],
    #     "input_ids": [],
    #     "attention_mask": []
    # }
    # Compute sentiment score
    # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)

    # Reward model IPC coms
    # test.split("Original Sentence: ")[-1].split('###Assistant:')[0].strip()
    # tempQuery = [
    #     Qitem.split("Original Sentence: ")[-1].split("###Assistant:")[0].strip()
    #     for Qitem in batch["query"]
    # ]

    submit_data = {
        "query": batch["raw"],
        "response": [x.strip() for x in batch["response"]],
        "label": batch["label"],
    }
    # print(batch["response"])
    res = None
    for retry in range(max_retries):
        response = requests.post(url, json=submit_data)

        if response.status_code == 200:
            res = response.json()
            # print(json.dumps(res, indent=4))
            break
        else:
            print("Error submitting data:", response.status_code)
            print(f"Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
    else:
        print("Max retries exceeded. Could not submit data.")
    # end of IPC

    rewards = [torch.tensor(output - script_args.reward_baseline) for output in res]

    # chatglm
    queries: List[torch.Tensor] = []
    responses: List[torch.Tensor] = []
    for i in range(len(question_tensors)):
        query_length = (question_tensors[i] != tokenizer.pad_token_id).nonzero()[0].item()
        response_length = (response_tensors[i] != tokenizer.pad_token_id).nonzero()[-1].item() + 1
        queries.append(question_tensors[i][query_length:])
        if response_length < 2:
            responses.append(response_tensors[i].new_empty(2).fill_(tokenizer.eos_token_id))
        else:
            responses.append(response_tensors[i][:response_length])
    # chatglm

    # Run PPO step
    # stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    stats = ppo_trainer.step(queries, responses, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and step_count and step_count % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"t5_step_{step_count}", push_to_hub= False)
ppo_trainer.save_pretrained(script_args.output_dir + f"t5_final", push_to_hub= False)
