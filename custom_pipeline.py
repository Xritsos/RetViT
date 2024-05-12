# Test file
from matplotlib import pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import MultilabelF1Score, MultilabelHammingDistance
from sklearn.metrics import f1_score

import pandas as pd
import os

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
        #self.vit = AutoModelForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224',
                                                              num_labels=5,
                                                              problem_type="multi_label_classification",
                                                              id2label=id2label,
                                                              label2id=label2id,
                                                              ignore_mismatched_sizes=True)

        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []
        self.test_predictions = []
        self.test_labels = []
        # custom_head = nn.Sequential(
        #     nn.Linear(1536, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(1024, 768),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(64, 8)
        # )
        #
        # # Replace the classifier with your custom head
        # self.vit.classifier = custom_head

        for param in self.vit.parameters():
            param.requires_grad = False

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

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        correct = (predicted_labels == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.train_labels.append(labels.cpu())
        self.train_predictions.append(predicted_labels.cpu())
        # self.log("train loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.val_labels.append(labels.cpu())
        self.val_predictions.append(predicted_labels.cpu())
        # self.log("val loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.test_labels.append(labels.cpu())
        self.test_predictions.append(predicted_labels.cpu())
        # self.log("test loss", loss, on_epoch=True, prog_bar=True)
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

    def on_train_epoch_end(self):
        train_preds = np.concatenate([t.numpy() for t in self.train_predictions])
        train_labels = np.concatenate([t.numpy() for t in self.train_labels])

        for i in range(5):
            label_score = f1_score(train_labels[:, i], train_preds[:, i], zero_division=1)
            self.log(f"{i} train f1", label_score, on_epoch=True, prog_bar=True)

        self.train_predictions.clear()  # free memory
        self.train_labels.clear()

    def on_validation_epoch_end(self):
        val_preds = np.concatenate([t.numpy() for t in self.val_predictions])
        val_labels = np.concatenate([t.numpy() for t in self.val_labels])

        for i in range(5):
            label_score = f1_score(val_labels[:, i], val_preds[:, i])
            self.log(f"{i} val f1", label_score, on_epoch=True, prog_bar=True)
        self.val_predictions.clear()
        self.val_labels.clear()

    def on_test_epoch_end(self):
        test_preds = np.concatenate([t.numpy() for t in self.test_predictions])
        test_labels = np.concatenate([t.numpy() for t in self.test_labels])

        for i in range(5):
            label_score = f1_score(test_labels[:, i], test_preds[:, i])
            self.log(f"{i} test f1", label_score, on_epoch=True, prog_bar=True)
        self.test_predictions.clear()
        self.test_labels.clear()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_ds, val_ds, test_ds = utils.load_data()

# integer to label mapping
id2label = {0: "N", 1: "D", 2: "C", 3: "M", 4: "O"}
label2id = {label:id for id, label in id2label.items()}


# size = processor.size["height"]


train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=utils.collate_fn, batch_size=batch_size,
                              num_workers=4,
                              persistent_workers=True,
                              pin_memory=True)

val_dataloader = DataLoader(val_ds, collate_fn=utils.collate_fn, batch_size=batch_size,
                            num_workers=4,
                            persistent_workers=True,
                            pin_memory=True)

test_dataloader = DataLoader(test_ds, collate_fn=utils.collate_fn, batch_size=batch_size,
                             num_workers=4,
                             persistent_workers=True,
                             pin_memory=True)

# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=early_stopping_patience,
    strict=False,
    verbose=False,
    mode='min'
)


csv_logger = CSVLogger(save_dir='logs/', name='experiment_name', flush_logs_every_n_steps=1)

# checkpoint_path = '/home/g/gbotso/Desktop/Project/RetViT/logs/experiment_name/version_2/checkpoints/epoch=41-step=11718.ckpt'
#model = ViTLightningModule.load_from_checkpoint(checkpoint_path)
model = ViTLightningModule()

trainer = Trainer(accelerator='gpu', max_epochs=num_epochs,
                  callbacks=[early_stop_callback], logger=csv_logger, log_every_n_steps=5)

trainer.fit(model)

print("================ Testing... ======================")
trainer.test(model, ckpt_path='best')
print("==================================================")


utils.plot_results()
