import os
import gc
import copy
import time
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoConfig, AdamW, DebertaV2TokenizerFast
from transformers import DataCollatorWithPadding
from text_unidecode import unidecode
from typing import Tuple
import codecs
import warnings

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

cfg = {
    "seed": 42,

    # Model Configs
    # "model_name": "../input/deberta-v3-large/deberta-v3-large",
    "model_name": "microsoft/deberta-v3-large",
    "max_len": 1024,
    "num_classes": 3,

    # Train Configs
    "n_fold": 5,
    "train_batch_size": 4,
    "valid_batch_size": 4,
    "epochs": 2,
    "learning_rate": 1e-5,
    "min_lr": 1e-6,
    "weight_decay": 1e-2,
    "max_dropout_rate": 0.5,
    "num_dropout": 5,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "data_fraction": 1,
    "unfreeze": False,
    "scheduler": 'CosineAnnealingLR',
    "t_max": 500,

    # Dir Path
    # "input_path": "../input/feedback-prize-effectiveness",
    # "output_path": "./"
    "input_path": "./autodl-tmp/kaggle/input/feedback-prize-effectiveness",
    "output_path": "./autodl-tmp/kaggle/output"
}

cfg["tokenizer"] = DebertaV2TokenizerFast.from_pretrained(cfg['model_name'])

print()
for key,val in cfg.items():
    if key == 'tokenizer':
        continue
    print(f"{key}: {val}")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(cfg['seed'])


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
    essay_path = os.path.join(cfg['input_path'], "train", f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text

def get_elem_len(elem):
    return len(elem)

def transform(df):
    ids = []
    mask = []
    for _, row in df.iterrows():
        text = row['discourse_type'] + cfg["tokenizer"].sep_token + row['discourse_text'] + cfg["tokenizer"].sep_token + row['essay_text']
        inputs = cfg["tokenizer"].encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=cfg['max_len']
        )
        ids.append(inputs['input_ids'])
        mask.append(inputs['attention_mask'])

    df = df.drop(['discourse_type', 'discourse_text', 'essay_text'], axis=1)
    df = df.assign(input_ids = ids, attention_mask = mask)
    df['elem_len'] = df['input_ids'].apply(get_elem_len)

    return df

def convert_feature(df, keep_label_encoder=False):
    df['essay_text'] = df['essay_id'].apply(get_essay)
    df['discourse_type'] = df['discourse_type'].apply(lambda x: resolve_encodings_and_normalize(x))
    df['discourse_text'] = df['discourse_text'].apply(lambda x: resolve_encodings_and_normalize(x))
    df['essay_text'] = df['essay_text'].apply(lambda x: resolve_encodings_and_normalize(x))

    df = transform(df)

    encoder = LabelEncoder()
    df['discourse_effectiveness'] = encoder.fit_transform(df['discourse_effectiveness'])

    if keep_label_encoder:
        with open("le.pkl", "wb") as fp:
            joblib.dump(encoder, fp)
    return df


class FeedbackDataset(Dataset):
    def __init__(self, df):
        self.df = df

        self.df = self.df.sort_values(by=['elem_len'])

        self.input_ids = self.df['input_ids'].values
        self.attention_mask = self.df['attention_mask'].values
        self.targets = self.df['discourse_effectiveness'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'target': self.targets[index]
        }


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
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
        output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size, hiddendim_fc, num_layers=24, total_layers=25):
        super(AttentionPooling, self).__init__()
        self.num_layers = num_layers
        self.total_layers = total_layers
        self.hidden_size = hidden_size
        self.hiddendim_fc = hiddendim_fc

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(cfg['device'])
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(cfg['device'])

    def forward(self, all_hidden_states):
        hidden_states = torch.stack([all_hidden_states[i][:, 0] for i in range(self.total_layers - self.num_layers, self.total_layers)], dim=-1)
        hidden_states = hidden_states.transpose(1, 2)
        out = self.attention(hidden_states)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v

class LSTMPooling(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=24, total_layers=25):
        super(LSTMPooling, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.total_layers = total_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=False)

    def forward(self, all_hidden_states):
        hidden_states = torch.stack([all_hidden_states[i][:, 0] for i in range(self.total_layers - self.num_layers, self.total_layers)], dim=-1)
        hidden_states = hidden_states.transpose(1, 2)
        out, _ = self.lstm(hidden_states, None)
        out = out[:, -1, :]
        return out

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class MultiSampleDropout(nn.Module):
    def __init__(self, classifier, max_dropout_rate=cfg['max_dropout_rate'], num_samples=cfg['num_dropout']):
        super(MultiSampleDropout, self).__init__()
        self.dropouts = [nn.Dropout(p) for p in np.linspace(0.1, max_dropout_rate, num_samples)]
        self.classifier = classifier

    def forward(self, out):
        return torch.mean(torch.stack([self.classifier(dropout(out)) for dropout in self.dropouts], dim=0), dim=0)


class FeedbackModel(nn.Module):

    def __init__(self, model_name):
        super(FeedbackModel, self).__init__()
        self.model_config = AutoConfig.from_pretrained(model_name)
        self.model_config.update({"output_hidden_states": True})

        self.transformer = AutoModel.from_pretrained(model_name, config=self.model_config)
        self.pooler_hor = MeanPooling()  # -> (batch_size, 1024)
        self.pooler_ver = AttentionPooling(self.model_config.hidden_size, self.model_config.hidden_size)  # -> (batch_size, 1024)
        self.fc = nn.Linear(self.model_config.hidden_size * 2, cfg['num_classes'])
        self.multi_sample_dropout = MultiSampleDropout(classifier=self.fc)

        if cfg['unfreeze']:
            self.freeze()

    def forward(self, ids, mask):
        out = self.transformer(input_ids=ids, attention_mask=mask)
        out_hor = self.pooler_hor(out.last_hidden_state, mask)
        out_ver = self.pooler_ver(torch.stack(out.hidden_states))
        out = torch.cat((out_hor, out_ver), 1)
        outputs = self.multi_sample_dropout(out)
        return outputs

    # def __init__(self, model_name, lstm_hidden_size=256):
    #     super(FeedbackModel, self).__init__()
    #     self.model_config = AutoConfig.from_pretrained(model_name)
    #     self.model_config.update({"output_hidden_states": True})
    #
    #     self.transformer = AutoModel.from_pretrained(model_name, config=self.model_config)
    #     self.pooler = LSTMPooling(self.model_config.hidden_size, lstm_hidden_size)
    #     self.fc = nn.Linear(lstm_hidden_size, cfg['num_classes'])
    #     self.multi_sample_dropout = MultiSampleDropout(classifier=self.fc)
    #
    #     if cfg['unfreeze']:
    #         self.freeze()
    #
    # def forward(self, ids, mask):
    #     out = self.transformer(input_ids=ids, attention_mask=mask)
    #     all_hidden_states = torch.stack(out.hidden_states)
    #     out = self.pooler(all_hidden_states)
    #     outputs = self.multi_sample_dropout(out)
    #     return outputs

    # def __init__(self, model_name):
    #     super(FeedbackModel, self).__init__()
    #     self.model_config = AutoConfig.from_pretrained(model_name)
    #     self.model_config.update({"output_hidden_states": False})
    #
    #     self.transformer = AutoModel.from_pretrained(model_name, config=self.model_config)
    #     self.pooler = MeanPooling()
    #     self.fc = nn.Linear(self.model_config.hidden_size, cfg['num_classes'])
    #     self.multi_sample_dropout = MultiSampleDropout(classifier=self.fc)
    #
    #     if cfg['unfreeze']:
    #         self.freeze()
    #
    # def forward(self, ids, mask):
    #     out = self.transformer(input_ids=ids, attention_mask=mask)
    #     out = self.pooler(out.last_hidden_state, mask)
    #     outputs = self.multi_sample_dropout(out)
    #     return outputs

    def freeze(self):
        for parameter in self.transformer.embeddings.parameters():
            parameter.requires_grad = False
        for parameter in self.transformer.encoder.layer.parameters():
            parameter.requires_grad = False

    def unfreeze(self):
        for parameter in self.transformer.embeddings.parameters():
            parameter.requires_grad = True
        for parameter in self.transformer.encoder.layer.parameters():
            parameter.requires_grad = True


def criterion(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


def fetch_scheduler(optimizer):
    if cfg['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['t_max'],
                                                   eta_min=cfg['min_lr'])
    else:
        milestones = [epoch_size * e for e in range(1, cfg['epochs'] + 1)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    return scheduler


def prepare_loaders(fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = FeedbackDataset(df_train)
    valid_dataset = FeedbackDataset(df_valid)

    # collate_fn = DataCollatorWithPadding(tokenizer=cfg['tokenizer'])
    collate_fn = Collate(cfg['tokenizer'])

    train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], collate_fn=collate_fn,
                              num_workers=8, shuffle=True, pin_memory=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['valid_batch_size'], collate_fn=collate_fn,
                              num_workers=8, shuffle=False, pin_memory=False)

    return train_loader, valid_loader


# Load Datset
df = pd.read_csv(os.path.join(cfg['input_path'], "train.csv"))
# df = df.sample(frac=cfg['data_fraction']).reset_index(drop=True)
df = convert_feature(df, keep_label_encoder=True)

epoch_size = int(len(df) / cfg['n_fold'] * (cfg['n_fold'] - 1) / cfg['train_batch_size'])

# Create Fold
gkf = GroupKFold(n_splits=cfg['n_fold'])
for fold, ( _, val_) in enumerate(gkf.split(X=df, groups=df.essay_id)):
    df.loc[val_, "kfold"] = int(fold)

df["kfold"] = df["kfold"].astype(int)
df.groupby('kfold')['discourse_effectiveness'].value_counts()


def save_model(model, history, fold):
    print(f"Validation Loss Improved to {history['Best Valid Loss'][0]}")
    best_model_wts = copy.deepcopy(model.state_dict())
    PATH = os.path.join(cfg['output_path'], f"Loss-Fold-{fold}.bin")
    torch.save(model.state_dict(), PATH)
    print(f"Model Saved")
    print()

def train_one_epoch(model, optimizer, scheduler, dataloader, epoch, history, fold, device=cfg['device']):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        # validate
        if cfg['scheduler'] == 'CosineAnnealingLR':
            if (step + epoch_size * (epoch-1)) % cfg['t_max'] == 0 and ((step + epoch_size * (epoch-1)) // cfg['t_max']) % 2 != 0:
                history['Train Loss'].append(epoch_loss)
                _ = valid_one_epoch(model, valid_loader, epoch, history, fold)
                model.train()

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])

    history['Train Loss'].append(epoch_loss)

    gc.collect()
    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model, dataloader, epoch, history, fold, device=cfg['device']):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    print('\n', 'validating...', '\n')

    for step, data in enumerate(dataloader):
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

    history['Valid Loss'].append(epoch_loss)

    # save model
    if (epoch_loss < history['Best Valid Loss'][0]):
        history['Best Valid Loss'][0] = epoch_loss
        save_model(model, history, fold)

    gc.collect()
    return epoch_loss

def start_train(model, optimizer, scheduler, device, num_epochs, fold):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    tic = time.time()

    history = defaultdict(list)
    history['Best Valid Loss'] = [np.inf]

    for epoch in range(1, num_epochs + 1):
        print(f"====== Epoch: {epoch} ======")
        gc.collect()

        # unfreeze pre-trained model from epoch 2
        if epoch == 2 and cfg['unfreeze']:
            model.unfreeze()

        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, train_loader, epoch, history, fold)

        val_epoch_loss = valid_one_epoch(model, valid_loader, epoch, history, fold)

    toc = time.time()
    time_elapsed = toc - tic

    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Train Losses:", history['Train Loss'])
    print("Valid Losses:", history['Valid Loss'])
    print("Best Loss: {:.4f}".format(history['Best Valid Loss'][0]))

    return history

# Go!!!
for fold in range(1, cfg['n_fold'] + 1):
# for fold in range(4, 5):
    print()
    print(f"======== Fold: {fold} ========")

    # create dataloaders
    train_loader, valid_loader = prepare_loaders(fold=fold - 1)

    model = FeedbackModel(cfg['model_name'])
    model.to(cfg['device'])

    # Define Optimizer and Scheduler
#     optimizer_parameters = [
#             {
#                 "params": [p for n, p in model.named_parameters() if 'transformer' not in n]
#             },
#             {
#                 "params": [p for n, p in model.named_parameters() if 'transformer' in n],
#                 "lr": cfg['min_lr']
#             }
#     ]
#     optimizer = AdamW(optimizer_parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    optimizer = AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = fetch_scheduler(optimizer)

    history = start_train(model, optimizer, scheduler, device=cfg['device'], num_epochs=cfg['epochs'], fold=fold)

    del model, history, train_loader, valid_loader
    gc.collect()
    print()
