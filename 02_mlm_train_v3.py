from functools import partial
from itertools import chain
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, Trainer, TrainingArguments
from tqdm import tqdm
import os
import random
from text_unidecode import unidecode
from typing import Tuple
import codecs
import warnings
import math
from transformers.utils import logging
from transformers import AdamW, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
import datasets
import re

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_info()
logger = logging.get_logger(__name__)
logger.info("INFO")
logger.warning("WARN")

INPUT_DIR = "./autodl-tmp/kaggle/input/feedback-prize-2021/"
OUTPUT_DIR = "./autodl-tmp/kaggle/output/mlm/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class CFG:
    seed = 42
    model_name = 'allenai/longformer-large-4096'
    epochs = 10
    batch_size = 2
    lr = 5e-6
    weight_decay = 1e-1
    max_len = 2048
    mask_prob = 0.15  # perc of tokens to convert to mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_multiple = 512


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG.seed)


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start: error.end].encode("utf-8"), error.end

def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start: error.end].decode("cp1252"), error.end

# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
            .decode("utf-8", errors="replace_decoding_with_cp1252")
            .encode("cp1252", errors="replace_encoding_with_utf8")
            .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

def get_essay_text(sample, data_dir):
    id_ = sample["essay_id"]
    with open(data_dir + "train/" + f"{id_}.txt", "r") as fp:
        sample["essay_text"] = resolve_encodings_and_normalize(fp.read())
    return sample


train_df = pd.read_csv(INPUT_DIR + "train.csv")
train_df.rename(columns={"id": "essay_id"}, inplace=True)

essay_text_ds = datasets.Dataset.from_dict({"essay_id": train_df.essay_id.unique()})
essay_text_ds = essay_text_ds.map(
        partial(get_essay_text, data_dir=INPUT_DIR),
        num_proc=1,
        batched=False,
        desc="Loading text files",
)
essay_text_df = essay_text_ds.to_pandas()

train_df["discourse_text"] = [resolve_encodings_and_normalize(x) for x in train_df["discourse_text"]]
train_df = train_df.merge(essay_text_df, on="essay_id", how="left")
del essay_text_df


disc_types = [
    "Claim",
    "Concluding Statement",
    "Counterclaim",
    "Evidence",
    "Lead",
    "Position",
    "Rebuttal",
]
cls_tokens_map = {label: f"[CLS_{label.upper()}]" for label in disc_types}
end_tokens_map = {label: f"[END_{label.upper()}]" for label in disc_types}

label2id = {
    "Adequate": 0,
    "Effective": 1,
    "Ineffective": 2,
}

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)
tokenizer.add_special_tokens(
    {"additional_special_tokens": list(cls_tokens_map.values()) + list(end_tokens_map.values())}
)
tokenizer.save_pretrained(OUTPUT_DIR + 'tokenizer/')

cls_id_map = {
    label: tokenizer.encode(tkn)[1] for label, tkn in cls_tokens_map.items()
}


def find_positions(sample):
    text = sample["essay_text"][0]

    # keeps track of what has already
    # been located
    min_idx = 0

    # stores start and end indexes of discourse_texts
    idxs = []

    for dt in sample["discourse_text"]:
        # calling strip is essential
        matches = list(re.finditer(re.escape(dt.strip()), text))

        # If there are multiple matches, take the first one
        # that is past the previous discourse texts.
        if len(matches) > 1:
            for m in matches:
                if m.start() >= min_idx:
                    break
        # If no matches are found
        elif len(matches) == 0:
            idxs.append([-1])  # will filter out later
            continue
            # If one match is found
        else:
            m = matches[0]

        idxs.append([m.start(), m.end()])

        min_idx = m.start()

    return idxs


def tokenize(sample):
    sample["idxs"] = find_positions(sample)

    text = sample["essay_text"][0]
    chunks = []
    prev = 0

    zipped = zip(
        sample["idxs"],
        sample["discourse_type"]
    )
    for idxs, disc_type in zipped:
        # when the discourse_text wasn't found
        if idxs == [-1]:
            continue

        s, e = idxs

        # if the start of the current discourse_text is not
        # at the end of the previous one.
        # (text in between discourse_texts)
        if s != prev:
            chunks.append(text[prev:s])
            prev = s

        # if the start of the current discourse_text is
        # the same as the end of the previous discourse_text
        if s == prev:
            chunks.append(cls_tokens_map[disc_type])
            chunks.append(text[s:e])
            chunks.append(end_tokens_map[disc_type])

        prev = e

    tokenized = tokenizer(
        " ".join(chunks),
        padding='max_length',
        truncation=True,
        max_length=CFG.max_len,
    )

    return tokenized


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=CFG.mask_prob)

train_grouped_df = train_df.groupby(["essay_id"]).agg(list)

train_ds = datasets.Dataset.from_pandas(train_grouped_df)
train_ds = train_ds.map(
                tokenize,
                batched=False,
                num_proc=1,
                desc="Tokenizing",
        )

keep_cols = {"input_ids", "attention_mask"}
train_dataset = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])

dataset_split = train_dataset.train_test_split(test_size=0.1)


logging_steps = len(dataset_split["train"]) // CFG.batch_size
model_name = CFG.model_name.split("/")[-1]

training_args = TrainingArguments(
    output_dir= OUTPUT_DIR,
    overwrite_output_dir=False,
    num_train_epochs=CFG.epochs,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=CFG.lr,
    weight_decay=CFG.weight_decay,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size,
    push_to_hub=False,
    fp16=True,
    logging_steps=logging_steps,
    report_to="none",
)

model = AutoModelForMaskedLM.from_pretrained(CFG.model_name)
model.resize_token_embeddings(len(tokenizer))
freeze(model.longformer.embeddings)
freeze(model.longformer.encoder.layer[:12])
model.to(CFG.device)

num_train_steps = int(len(dataset_split["train"]) / CFG.batch_size * CFG.epochs)
optimizer = AdamW(model.parameters(), lr=CFG.lr)
# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps, num_cycles=0.5)
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
)

trainer.train()


# eval_results = trainer.evaluate()
# print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# trainer.save_model()
