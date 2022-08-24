import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from transformers import DebertaV2TokenizerFast, AutoModelWithLMHead
from transformers import AdamW
from tqdm import tqdm
import os
import random
from text_unidecode import unidecode
from typing import Tuple
import codecs
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CFG:
    seed = 42
    model_name = 'microsoft/deberta-v3-large'
    epochs = 3
    batch_size = 2
    lr = 1e-6
    weight_decay = 1e-6
    max_len = 1024
    mask_prob = 0.15  # perc of tokens to convert to mask
    n_accumulate = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_path = "./autodl-tmp/kaggle/input/feedback-prize-2021"


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG.seed)


tokenizer = DebertaV2TokenizerFast.from_pretrained(CFG.model_name)

special_tokens = tokenizer.encode_plus('[CLS] [SEP] [MASK] [PAD]',
                                        add_special_tokens = False,
                                        return_tensors='pt')
special_tokens = torch.flatten(special_tokens["input_ids"])

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

def get_essay(essay_id):
    essay_path = os.path.join(CFG.input_path, "train", f"{essay_id}.txt")
    with open(essay_path) as f:
        text = f.readlines()
        full_text = ' '.join([x for x in text])
        essay_text = ' '.join([x for x in full_text.split()])
    return essay_text

def getMaskedLabels(input_ids):
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < CFG.mask_prob)
    # Preventing special tokens to get replace by the [MASK] token
    for special_token in special_tokens:
        token = special_token.item()
        mask_arr *= (input_ids != token)
    selection = torch.flatten(mask_arr[0].nonzero()).tolist()
    input_ids[selection] = 128000

    return input_ids

class MLMDataset:
    def __init__(self, data, tokenizer):
        self.data = data['text']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokenized_data = self.tokenizer.encode_plus(
                            text,
                            max_length = CFG.max_len,
                            truncation = True,
                            padding = 'max_length',
                            add_special_tokens = True,
                            return_tensors = 'pt'
                        )
        input_ids = torch.flatten(tokenized_data.input_ids)
        attention_mask = torch.flatten(tokenized_data.attention_mask)
        labels = getMaskedLabels(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


train_data = []
TRAIN_DIR = os.path.join(CFG.input_path, "train")
for file in tqdm(os.listdir(TRAIN_DIR)):
    with open(os.path.join(TRAIN_DIR, file), 'r') as f:
        # text = f.readlines()
        # full_text = ' '.join([x for x in text])
        # essay_text = ' '.join([x for x in full_text.split()])
        # essay_text = resolve_encodings_and_normalize(essay_text)
        essay_text = f.read()
        train_data.append({'text': essay_text, 'id': file[:-4]})

essay_data = pd.DataFrame(train_data)

dataset = MLMDataset(essay_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True)


model = AutoModelWithLMHead.from_pretrained(CFG.model_name)

optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)


def train_loop(model, device):
    model.train()
    batch_losses = []
    loop = tqdm(dataloader, leave=True)
    for batch_num, batch in enumerate(loop):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        batch_loss = loss / CFG.n_accumulate
        batch_losses.append(batch_loss.item())

        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=batch_loss.item())
        batch_loss.backward()

        if batch_num % CFG.n_accumulate == 0 or batch_num == len(dataloader) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            model.zero_grad()

    return np.mean(batch_losses)


device = CFG.device
model.to(device)
history = []
model.gradient_checkpointing_enable()
print(f"Gradient Checkpointing: {model.is_gradient_checkpointing}")

for epoch in range(CFG.epochs):
    loss = train_loop(model, device)
    history.append(loss)
    print(f"Loss: {loss}")

    torch.save(model.state_dict(), f"./deberta_mlm/Loss-Epoch-{epoch}.bin")
