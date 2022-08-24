import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
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


warnings.filterwarnings("ignore")

gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIR = "./autodl-tmp/kaggle/input/feedback-prize-effectiveness/"
OUTPUT_DIR = "./autodl-tmp/kaggle/output/z_baseline/"
# model_path = "microsoft/deberta-v3-large"
# model_path = "google/bigbird-roberta-large"
# model_path = "xlnet-large-cased"

class CFG:
    # model = "microsoft/deberta-v3-large"
    # model = "google/bigbird-roberta-large"
    # model = "xlnet-large-cased"
    seed = 42
    max_len = 512
    batch_size = 6
    epochs = 1
    n_fold = 5
    trn_fold = [0]
    lr = 1e-5
    weight_decay = 1e-2
    dropout = 0.1
    n_accumulate = 1
    eps = 1e-6
    betas = (0.9, 0.999)
    scheduler = 'cosine'
    num_workers = 8
    train = True
    num_warmup_steps = 0
    num_cycles = 0.5
    freezing = True
    layer_cls = -1


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


def get_essay(essay_id, is_train=True):
    parent_path = INPUT_DIR + 'train' if is_train else INPUT_DIR + 'test'
    essay_path = os.path.join(parent_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text


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


def criterion_val(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


def criterion_train(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


def get_score(outputs, labels):
    outputs = F.softmax(torch.tensor(outputs)).numpy()
    score = log_loss(labels, outputs)
    return round(score, 5)


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


train = pd.read_csv(INPUT_DIR + 'train.csv')

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenizer.save_pretrained(OUTPUT_DIR + 'tokenizer/')
CFG.tokenizer = tokenizer

train['essay_text'] = train['essay_id'].apply(lambda x: get_essay(x, is_train=True))

# fix incorrect discourse_text
train['discourse_text'][293] = 'Cl' + train['discourse_text'][293]
train['discourse_text'][790] = 'T' + train['discourse_text'][790]
train['discourse_text'][879] = 'I' + train['discourse_text'][879]
train['discourse_text'][2828] = 'w' + train['discourse_text'][2828]
train['discourse_text'][4793] = 'i' + train['discourse_text'][4793]
train['discourse_text'][8093] = 'I' + train['discourse_text'][8093]
train['discourse_text'][9202] = 'l' + train['discourse_text'][9202]
train['discourse_text'][9790] = 'I' + train['discourse_text'][9790]
train['discourse_text'][14054] = 'i' + train['discourse_text'][14054]
train['discourse_text'][14387] = 's' + train['discourse_text'][14387]
train['discourse_text'][15188] = 'i' + train['discourse_text'][15188]
train['discourse_text'][15678] = 'I' + train['discourse_text'][15678]
train['discourse_text'][16065] = 'f' + train['discourse_text'][16065]
train['discourse_text'][16084] = 'I' + train['discourse_text'][16084]
train['discourse_text'][16255] = 'T' + train['discourse_text'][16255]
train['discourse_text'][17096] = 'I' + train['discourse_text'][17096]
train['discourse_text'][17261] = 't' + train['discourse_text'][17261]
train['discourse_text'][18691] = 'I' + train['discourse_text'][18691]
train['discourse_text'][19967] = 't' + train['discourse_text'][19967]
train['discourse_text'][20186] = 'b' + train['discourse_text'][20186]
train['discourse_text'][20264] = 'I' + train['discourse_text'][20264]
train['discourse_text'][20421] = 'i' + train['discourse_text'][20421]
train['discourse_text'][20870] = 'h' + train['discourse_text'][20870]
train['discourse_text'][22064] = 't' + train['discourse_text'][22064]
train['discourse_text'][22793] = 'I' + train['discourse_text'][22793]
train['discourse_text'][22962] = 'W' + train['discourse_text'][22962]
train['discourse_text'][23990] = 'f' + train['discourse_text'][23990]
train['discourse_text'][24085] = 'w' + train['discourse_text'][24085]
train['discourse_text'][25330] = 'a' + train['discourse_text'][25330]
train['discourse_text'][25446] = 'i' + train['discourse_text'][25446]
train['discourse_text'][25667] = 'S' + train['discourse_text'][25667]
train['discourse_text'][25869] = 'I' + train['discourse_text'][25869]
train['discourse_text'][26172] = 'i' + train['discourse_text'][26172]
train['discourse_text'][26284] = 'I' + train['discourse_text'][26284]
train['discourse_text'][26289] = 't' + train['discourse_text'][26289]
train['discourse_text'][26322] = 't' + train['discourse_text'][26322]
train['discourse_text'][26511] = 't' + train['discourse_text'][26511]
train['discourse_text'][27763] = 'I' + train['discourse_text'][27763]
train['discourse_text'][28262] = 'P' + train['discourse_text'][28262]
train['discourse_text'][29164] = 'bu' + train['discourse_text'][29164]
train['discourse_text'][29519] = 'e' + train['discourse_text'][29519]
train['discourse_text'][29532] = 't' + train['discourse_text'][29532]
train['discourse_text'][29571] = 'A' + train['discourse_text'][29571]
train['discourse_text'][29621] = 't' + train['discourse_text'][29621]
train['discourse_text'][30791] = 'E' + train['discourse_text'][30791]
train['discourse_text'][30799] = 'T' + train['discourse_text'][30799]
train['discourse_text'][31519] = 't' + train['discourse_text'][31519]
train['discourse_text'][31597] = 't' + train['discourse_text'][31597]
train['discourse_text'][31992] = 'T' + train['discourse_text'][31992]
train['discourse_text'][32086] = 'I' + train['discourse_text'][32086]
train['discourse_text'][32204] = 'c' + train['discourse_text'][32204]
train['discourse_text'][32341] = 'becaus' + train['discourse_text'][32341]
train['discourse_text'][33246] = 'A' + train['discourse_text'][33246]
train['discourse_text'][33819] = 'W' + train['discourse_text'][33819]
train['discourse_text'][34023] = 'i' + train['discourse_text'][34023]
train['discourse_text'][35467] = 'b' + train['discourse_text'][35467]
train['discourse_text'][35902] = 'i' + train['discourse_text'][35902]

SEP = tokenizer.sep_token
train['text'] = train['discourse_type'] + ' ' + train['discourse_text'] + SEP + train['essay_text']
# train['text'] = train['text'].apply(lambda x: resolve_encodings_and_normalize(x))

gkf = StratifiedGroupKFold(n_splits=CFG.n_fold)
for fold, (train_id, val_id) in enumerate(gkf.split(X=train, y=train.discourse_effectiveness, groups=train.essay_id)):
    train.loc[val_id, "fold"] = int(fold)
train["fold"] = train["fold"].astype(int)
train.groupby('fold')['discourse_effectiveness'].value_counts()

train['label'] = train['discourse_effectiveness'].map({'Ineffective': 0, 'Adequate': 1, 'Effective': 2})

class FeedBackDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.max_len = CFG.max_len
        self.text = df['text'].values
        self.tokenizer = CFG.tokenizer
        self.targets = df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len
        )
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': self.targets[index]
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
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in
                                   output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in
                                   output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output

collate_fn = Collate(CFG.tokenizer, isTrain=True)


class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update({"output_hidden_states": True})
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.fc = nn.Linear(self.config.hidden_size, 3)
        if CFG.freezing:
            # freeze(self.model.embeddings)
            # freeze(self.model.encoder.layer[:12])
            # freeze(self.model.word_embedding)  # xlnet
            # freeze(self.model.layer[:12])  # xlnet

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask)
        all_hidden_states = torch.stack(out.hidden_states)
        cls_embeddings = all_hidden_states[CFG.layer_cls, :, 0]
        outputs = self.fc(cls_embeddings)
        return outputs


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


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


def train_one_epoch(model, optimizer, scheduler, dataloader, epoch, swa_start, swa_scheduler, swa_model):
    model.train()

    dataset_size = 0
    running_loss = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        outputs = model(ids, mask)
        loss = criterion_train(outputs, targets)

        # accumulate
        loss = loss / CFG.n_accumulate
        loss.backward()
        if (step + 1) % CFG.n_accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()

            if step >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch+1, Epoch_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])

    # Update bn statistics for the swa_model at the end
    torch.optim.swa_utils.update_bn(dataloader, swa_model)

    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0

    pred = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)
        outputs = model(ids, mask)
        loss = criterion_val(outputs, targets)
        pred.append(outputs.to('cpu').numpy())

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch+1, Epoch_Loss=epoch_loss)

    pred = np.concatenate(pred)

    return epoch_loss, pred


def train_loop(fold):

    LOGGER.info(f'-------------fold:{fold} training-------------')

    train_data = train[train.fold != fold].reset_index(drop=True)
    valid_data = train[train.fold == fold].reset_index(drop=True)
    valid_labels = valid_data.label.values

    trainDataset = FeedBackDataset(train_data)
    validDataset = FeedBackDataset(valid_data)

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
    swa_model = AveragedModel(model)

    def get_optimizer_params(model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': CFG.lr, 'weight_decay': CFG.weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': CFG.lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model)
    optimizer = AdamW(optimizer_parameters, lr=CFG.lr, eps=CFG.eps, betas=CFG.betas)
    num_train_steps = int(len(train_data) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    swa_start = int(num_train_steps * 0.795)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-6)

    # loop
    best_score = 100

    for epoch in range(CFG.epochs):
        start_time = time.time()

        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, train_loader, epoch, swa_start, swa_scheduler, swa_model)
        # Use swa_model to make predictions on test data
        valid_epoch_loss, pred = valid_one_epoch(swa_model, valid_loader, epoch)

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {train_epoch_loss:.4f}  avg_val_loss: {valid_epoch_loss:.4f}  time: {elapsed:.0f}s')

        score = get_score(pred, valid_labels)
        LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}')

        if score < best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': swa_model.state_dict(),
                        'predictions': pred},
                       OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                             map_location=torch.device('cpu'))['predictions']
    valid_data['pred_0'] = predictions[:, 0]
    valid_data['pred_1'] = predictions[:, 1]
    valid_data['pred_2'] = predictions[:, 2]

    torch.cuda.empty_cache()
    gc.collect()

    return valid_data


if __name__ == '__main__':

    def get_result(oof_df):
        labels = oof_df['label'].values
        preds = oof_df[['pred_0', 'pred_1', 'pred_2']].values.tolist()
        score = get_score(preds, labels)
        LOGGER.info(f'Score: {score:<.4f}')

    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR + 'oof_df.pkl')
        oof_df.to_csv(OUTPUT_DIR + f'oof_df.csv', index=False)
