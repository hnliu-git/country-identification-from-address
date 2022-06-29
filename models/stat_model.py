"""
Statistical model for the country identification
"""

import torch
import pandas as pd
import pytorch_lightning as pl
import torch.utils.data

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.types import _PATH

import os
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins import CheckpointIO
from transformers import get_linear_schedule_with_warmup

from pytorch_lightning.utilities.cloud_io import get_filesystem


class TextDFData(torch.utils.data.Dataset):

    @staticmethod
    def collate_fn(batch):

        batch_dict = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }

        for X, y in batch:
            batch_dict['input_ids'].append(torch.LongTensor(X['input_ids']))
            batch_dict['attention_mask'].append(torch.LongTensor(X['attention_mask']))
            batch_dict['labels'].append(torch.LongTensor([y]))

        batch_dict = {k: torch.stack(v, dim=0) for k, v in batch_dict.items()}
        batch_dict['labels'] = torch.squeeze(batch_dict['labels'])

        return batch_dict

    def __init__(self, countries, tokenizer, split, max_len = 128):
        super().__init__()
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.code2id = {code: i for i, code in enumerate(countries)}

        df = pd.DataFrame()
        for country in countries:
            df= pd.concat([df, pd.read_csv(f'data/{split}/{country}.csv', sep='\t')])
        df.reset_index(drop=True, inplace=True)

        self.df = df

    def __getitem__(self, idx):
        address = self.df.iloc[idx]['address']
        country = self.df.iloc[idx]['country']

        X = self.tokenizer(address,truncation=True, padding='max_length', max_length=self.max_len)
        y = self.code2id[country]

        return X, y

    def __len__(self):
        return len(self.df)

    def steps_per_epoch(self, batch_size):
        return len(self.df) // batch_size


class HgCkptIO(CheckpointIO):

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        '''Save the fine-tuned model in a hugging-face style.
        Args:
            checkpoint: ckpt, but only key 'hg_model' matters
            path: path to save the ckpt
            storage_options: not used
        '''
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint['hg_model'].save_pretrained(path)

    def load_checkpoint(self, path: _PATH, storage_options: Optional[Any] = None) -> Dict[str, Any]:
        pass

    def remove_checkpoint(self, path: _PATH) -> None:
        """Remove checkpoint file from the filesystem.
        Args:
            path: Path to checkpoint
        """
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)


class StatModel(LightningModule):

    def __init__(self, countries, model, num_training_steps):
        super().__init__()

        self.num_classes = len(countries)
        self.num_training_steps = num_training_steps
        self.code2id = {code: i for i, code in enumerate(countries)}
        self.id2code = {i: code for i, code in enumerate(countries)}

        self.model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=self.num_classes)

        # Metrics
        self.acc = torchmetrics.Accuracy(num_classes=self.num_classes)
        self.f1 = torchmetrics.F1Score(num_classes=self.num_classes)

    def forward(self, batch):
        return self.model(**batch)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        parameters = [(n, p) for n, p in self.named_parameters()]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": 5e-5,
            },
            {
                "params": [p for n, p in parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=self.num_training_steps,
            num_warmup_steps=self.num_training_steps // 10,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, idx):
        loss = self(batch).loss
        self.log("pred:nll", loss, logger=True)
        return loss

    def validation_step(self, batch, idx):
        labels = batch['labels']
        out = self(batch)
        pred = torch.argmax(out.logits, dim=1)
        self.f1(pred, labels)
        self.acc(pred, labels)

        return {'val_loss': out.loss}

    def validation_epoch_end(self, outputs) -> None:
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        self.log('val_acc', self.acc, logger=True)
        self.log('val_f1', self.f1, logger=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        """
            For the customized CheckpointIO
        """
        checkpoint['hg_model'] = self.model