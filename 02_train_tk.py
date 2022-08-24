import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import gc
import time
import math
import random
import warnings
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from text_unidecode import unidecode
from typing import Tuple
import codecs
from torch.optim.swa_utils import AveragedModel, SWALR
import re


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIR = "./autodl-tmp/kaggle/input/feedback-prize-effectiveness/"
OUTPUT_DIR = "./autodl-tmp/kaggle/output/zt_new/"
# model_path = "microsoft/deberta-v3-large"
# model_path = "allenai/longformer-large-4096"
# model_path = "microsoft/deberta-v2-xlarge"
model_path = "./autodl-tmp/kaggle/output/mlm/"

class CFG:
    model = "deberta-v3-large"
    seed = 42
    max_len = 2048
    batch_size = 1
    epochs = 2
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    lr = 1e-5
    weight_decay = 5e-1
    min_lr = 2e-6
    n_accumulate = 1
    eps = 1e-6
    betas = (0.9, 0.999)
    scheduler = 'cosine'
    num_workers = 8
    num_warmup_steps = 0
    num_cycles = 0.5
    freezing = True
    layer_cls = -1
    fgm = False
    awp = False
    awp_start = 0.75
    awp_loss = 0.6
    swa = False
    swa_start = 0.75
    save_mode = True

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_logger(filename=OUTPUT_DIR + 'train'):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=CFG.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

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

def get_essay(essay_id, is_train=True):
    parent_path = INPUT_DIR + 'train' if is_train else INPUT_DIR + 'test'
    essay_path = os.path.join(parent_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text

def criterion_train(outputs, labels, mask):
    loss_fct = nn.CrossEntropyLoss()
    # Only keep active parts of the loss
    active_loss = mask.view(-1) == 1
    active_logits = outputs.view(-1, 3)
    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
    )
    loss = loss_fct(active_logits, active_labels)
    return loss

def criterion_eval(outputs, labels, mask):
    loss_fct = nn.CrossEntropyLoss()
    # Only keep active parts of the loss
    active_loss = mask.view(-1) == 1
    active_logits = outputs.view(-1, 3)
    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
    )
    loss = loss_fct(active_logits, active_labels)
    return loss

def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class AWP:
    def __init__(self, model, adv_param: str="weight", adv_lr: float=1.0, adv_eps: float=0.02):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        self._save()
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )



train = pd.read_csv(INPUT_DIR + 'train.csv')

train['essay_text'] = train['essay_id'].apply(lambda x: get_essay(x, is_train=True))
train["discourse_text"] = [resolve_encodings_and_normalize(x) for x in train["discourse_text"]]
train["essay_text"] = [resolve_encodings_and_normalize(x) for x in train["essay_text"]]

gkf = GroupKFold(n_splits=CFG.n_fold)
for fold, (train_id, val_id) in enumerate(gkf.split(X=train, y=train.discourse_effectiveness, groups=train.essay_id)):
    train.loc[val_id, "fold"] = int(fold)
train["fold"] = train["fold"].astype(int)
train.groupby('fold')['discourse_effectiveness'].value_counts()

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


tokenizer = AutoTokenizer.from_pretrained(model_path + 'tokenizer', use_fast=True)
# tokenizer.add_special_tokens(
#     {"additional_special_tokens": list(cls_tokens_map.values()) + list(end_tokens_map.values())}
# )
tokenizer.save_pretrained(OUTPUT_DIR + 'tokenizer/')
CFG.tokenizer = tokenizer


cls_id_map = {
    label: tokenizer.encode(tkn)[1] for label, tkn in cls_tokens_map.items()
}

def find_positions(text, discourse_text):

    # keeps track of what has already
    # been located
    min_idx = 0

    # stores start and end indexes of discourse_texts
    idxs = []

    for dt in discourse_text:
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



class FeedBackDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.discourse_type = df['discourse_type'].values
        self.discourse_text = df['discourse_text'].values
        self.discourse_effectiveness = df['discourse_effectiveness'].values
        self.essay_text = df['essay_text'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.essay_text[index][0]
        discourse_text = self.discourse_text[index]

        chunks = []
        labels = []
        prev = 0

        zipped = zip(
            find_positions(text, discourse_text),
            self.discourse_type[index],
            self.discourse_effectiveness[index],
        )

        for idxs, disc_type, disc_effect in zipped:
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
            labels.append(label2id[disc_effect])

        tokenized = tokenizer(
            " ".join(chunks),
            truncation=True,
            add_special_tokens=True,
            max_length=CFG.max_len,
        )

        # at this point, labels is not the same shape as input_ids.
        # The following loop will add -100 so that the loss function
        # ignores all tokens except CLS tokens

        # idx for labels list
        idx = 0
        final_labels = []
        for id_ in tokenized["input_ids"]:
            # if this id belongs to a CLS token
            if id_ in cls_id_map.values():
                final_labels.append(labels[idx])
                idx += 1
            else:
                # -100 will be ignored by loss function
                final_labels.append(-100)

        tokenized["labels"] = final_labels

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'target': tokenized["labels"]
        }


class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in
                               output["input_ids"]]
        output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)

        if self.isTrain:
            output["target"] = [s + (batch_max - len(s)) * [-100] for s in output["target"]]
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output

collate_fn = Collate(CFG.tokenizer, isTrain=True)


class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update({"output_hidden_states": True})
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        # self.model.resize_token_embeddings(len(tokenizer))
        self.fc = nn.Linear(self.config.hidden_size, 3)
        if CFG.freezing:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:12])

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask)
        all_hidden_states = torch.stack(out.hidden_states)
        cls_embeddings = all_hidden_states[CFG.layer_cls]
        outputs = self.fc(cls_embeddings)
        return outputs


def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
            num_cycles=cfg.num_cycles
        )
    return scheduler

def get_optimizer_params(model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': CFG.lr, 'weight_decay': CFG.weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': CFG.lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def train_one_epoch(model, optimizer, scheduler, dataloader, epoch, swa_scheduler=None, swa_model=None, fgm=None, awp=None):
    model.train()

    dataset_size = 0
    running_loss = 0
    awp_on = False

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        outputs = model(ids, mask)
        loss = criterion_train(outputs, targets, mask)

        # accumulate
        loss = loss / CFG.n_accumulate
        loss.backward()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        # adverserial
        # if fgm:
        #     fgm.attack()
        #     outputs_adv = model(ids, mask)
        #     loss_adv = criterion_train(outputs_adv, targets, mask)
        #     loss_adv.backward()
        #     fgm.restore()

        if awp and step + len(dataloader) * epoch >= len(dataloader) * CFG.epochs * CFG.awp_start:
        #     # if epoch_loss < CFG.awp_loss:
        #     #     awp_on = True
        #     # if awp_on:
            awp.attack()
            outputs_awp = model(ids, mask)
            loss_awp = criterion_train(outputs_awp, targets, mask)
            loss_awp.backward()
            awp.restore()

        if (step + 1) % CFG.n_accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step()

            # if optimizer.param_groups[0]['lr'] > CFG.min_lr:
            #     scheduler.step()

            # if swa_model and step + len(dataloader) * epoch >= len(dataloader) * CFG.epochs * CFG.swa_start:
            #     swa_model.update_parameters(model)
            #     swa_scheduler.step()
            # else:
            #     scheduler.step()

        bar.set_postfix(Epoch=epoch+1, Epoch_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])

    # Update bn statistics for the swa_model at the end
    # if swa_model and len(dataloader) * (epoch + 1) > len(dataloader) * CFG.epochs * CFG.swa_start:
    #     torch.optim.swa_utils.update_bn(dataloader, swa_model)

    gc.collect()

    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model, dataloader, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)
        outputs = model(ids, mask)
        loss = criterion_eval(outputs, targets, mask)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch+1, Epoch_Loss=epoch_loss)

    return epoch_loss

def train_loop(fold):

    LOGGER.info(f'-------------fold:{fold} training-------------')

    train_data = train[train.fold != fold].reset_index(drop=True)
    valid_data = train[train.fold == fold].reset_index(drop=True)

    train_grouped_df = train_data.groupby(["essay_id"]).agg(list)
    valid_grouped_df = valid_data.groupby(["essay_id"]).agg(list)

    trainDataset = FeedBackDataset(train_grouped_df)
    validDataset = FeedBackDataset(valid_grouped_df)

    train_loader = DataLoader(trainDataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=True)
    valid_loader = DataLoader(validDataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=False)

    model = FeedBackModel(model_path)
    torch.save(model.config, OUTPUT_DIR + 'config.pth')
    model.to(device)

    if CFG.fgm:
        fgm = FGM(model)
    else:
        fgm = None

    if CFG.awp:
        awp = AWP(model)
    else:
        awp = None

    optimizer_parameters = get_optimizer_params(model)
    optimizer = AdamW(optimizer_parameters, lr=CFG.lr, eps=CFG.eps, betas=CFG.betas)
    num_train_steps = int(len(train_loader) * (CFG.epochs))
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    min_loss = 100

    if CFG.swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=1e-6)
        min_loss_swa = 100
    else:
        swa_model = None
        swa_scheduler = None

    for epoch in range(CFG.epochs):

        start_time = time.time()

        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, train_loader, epoch,
                                            swa_scheduler=swa_scheduler, swa_model=swa_model,
                                            fgm=fgm, awp=awp)
        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {train_epoch_loss:.4f}  time: {time.time() - start_time:.0f}s')

        valid_epoch_loss = valid_one_epoch(model, valid_loader, epoch)
        LOGGER.info(
            f'Epoch {epoch + 1} - avg_val_loss: {valid_epoch_loss:.4f}  time: {time.time() - start_time:.0f}s')

        if CFG.save_mode and valid_epoch_loss < min_loss:
            min_loss = valid_epoch_loss
            torch.save({'model': model.state_dict()},
                       OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
            LOGGER.info(f'Saved model with eval loss {min_loss}')

        if CFG.swa and len(train_loader) * (epoch + 1) > len(train_loader) * CFG.epochs * CFG.swa_start:
            valid_epoch_loss_swa = valid_one_epoch(swa_model, valid_loader, epoch)
            LOGGER.info(
                f'Epoch {epoch + 1} - avg_val_loss_swa: {valid_epoch_loss_swa:.4f}  time: {time.time() - start_time:.0f}s')
            if CFG.save_mode and valid_epoch_loss_swa < min_loss_swa:
                min_loss_swa = valid_epoch_loss_swa
                torch.save({'model': swa_model.state_dict()},
                           OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best_swa.pth")
                LOGGER.info(f'Saved swa model with swa eval loss {valid_epoch_loss_swa}')

    torch.cuda.empty_cache()
    gc.collect()


for fold in range(CFG.n_fold):
    if fold in CFG.trn_fold:
        train_loop(fold)
