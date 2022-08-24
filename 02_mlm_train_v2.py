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
    model_name = 'microsoft/deberta-v3-large'
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


train_data = []
TRAIN_DIR = INPUT_DIR + "train"
for file in tqdm(os.listdir(TRAIN_DIR)):
    with open(os.path.join(TRAIN_DIR, file), 'r') as f:
        essay_text = resolve_encodings_and_normalize(f.read())
        train_data.append({'text': essay_text, 'id': file[:-4]})
df_train = pd.DataFrame(train_data)
dataset = datasets.Dataset.from_pandas(df_train)
del df_train


tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)
tokenizer.save_pretrained(OUTPUT_DIR + 'tokenizer/')


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=CFG.mask_prob)


def tokenize(batched_data):
    tokenized = tokenizer(
        batched_data['text'],
        padding='max_length',
        truncation=True,
        max_length=CFG.max_len,
    )
    return tokenized

train_ds = dataset.map(
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
freeze(model.deberta.embeddings)
freeze(model.deberta.encoder.layer[:12])
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
