# Test file
from matplotlib import pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics.classification import MultilabelF1Score


import pandas as pd

# from model import ViTLightningModule

import pytorch_lightning as pl
from transformers import AutoModelForImageClassification, AdamW
from transformers import SwinConfig, ViTForImageClassification
import torch.nn as nn
from source import utils

seed_value = 42
torch.manual_seed(seed_value)

config_path = "configuration/config.csv"

learning_rate, weight_decay, batch_size, early_stopping_patience, num_epochs, criterion = utils.get_params(config_path)


class ViTLightningModule(pl.LightningModule):
    def __init__(self):
        super(ViTLightningModule, self).__init__()
        self.vit = AutoModelForImageClassification.from_pretrained('microsoft/swin-large-patch4-window12-384',
                                                              num_labels=8,
                                                              problem_type="multi_label_classification",
                                                              id2label=id2label,
                                                              label2id=label2id,
                                                              ignore_mismatched_sizes=True)


        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze head layers
        for param in self.vit.classifier.parameters():
            param.requires_grad = True

    # add this above for the multilabel case, plus more params values for multilabel
    #  problem_type="multi_label_classification",

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        loss = criterion(logits, labels.float())
        # predictions = logits.argmax(-1) # for cross entropy

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        correct = (predicted_labels == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]
        f1 = MultilabelF1Score(num_labels=8, average='weighted').to(device)
        f1 = f1(predicted_labels, labels)

        return loss, accuracy, f1

    def training_step(self, batch, batch_idx):
        loss, accuracy, f1 = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss, on_epoch=True, prog_bar=True)
        self.log("training F1 Score", f1, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, f1 = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True)
        self.log("validation F1 Score", f1, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, f1 = self.common_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test F1 Score", f1, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

    def test_dataloader(self):
        return test_dataloader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_ds, val_ds, test_ds = utils.load_data()

# integer to label mapping
id2label = {0: "N", 1: "D", 2: "G", 3: "C", 4: "A", 5: "H", 6: "M", 7: "O"}
label2id = {label:id for id, label in id2label.items()}


# size = processor.size["height"]


train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=utils.collate_fn, batch_size=batch_size,
                              num_workers=4)

val_dataloader = DataLoader(val_ds, collate_fn=utils.collate_fn, batch_size=batch_size,
                            num_workers=4)

test_dataloader = DataLoader(test_ds, collate_fn=utils.collate_fn, batch_size=batch_size,
                             num_workers=4)

# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=early_stopping_patience,
    strict=False,
    verbose=False,
    mode='min'
)

model = ViTLightningModule()

trainer = Trainer(accelerator='gpu', max_epochs=num_epochs,
                  callbacks=[early_stop_callback])

trainer.fit(model)

print()
print("================ Testing... ======================")
trainer.test(ckpt_path='best')
print("==================================================")
