# Test file
import torch
import numpy as np

from sklearn.metrics import f1_score, confusion_matrix
from torchmetrics.classification import MultilabelRankingAveragePrecision

import pytorch_lightning as pl
from transformers import AutoModelForImageClassification, AdamW, BeitForImageClassification,\
    LevitForImageClassification, DeiTForImageClassification, ImageGPTForImageClassification,\
    ResNetForImageClassification, ViTForImageClassification
from torch.utils.data import DataLoader

import config
import source.utils as utils


class ViTLightningModule(pl.LightningModule):
    def __init__(self):
        super(ViTLightningModule, self).__init__()

        #self.vit = AutoModelForImageClassification.from_pretrained('microsoft/swin-large-patch4-window12-384',
        self.vit = AutoModelForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224',
                                                              num_labels=config.num_labels,
                                                              problem_type="multi_label_classification",
                                                              id2label=config.id2label,
                                                              label2id=config.label2id,
                                                              ignore_mismatched_sizes=True)

        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []
        self.test_predictions = []
        self.test_labels = []
        self.epoch_train_loss = 0.0
        self.epoch_val_loss = 0.0
        self.train_steps = 0
        self.val_steps = 0

        self.train_ds, self.val_ds, self.test_ds = utils.load_data(dataset=config.dataset)

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


    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = config.criterion(logits, labels.float())

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        correct = (predicted_labels == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.train_labels.append(labels.cpu())
        self.train_predictions.append(predicted_labels.cpu())
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.val_labels.append(labels.cpu())
        self.val_predictions.append(predicted_labels.cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.test_labels.append(labels.cpu())
        self.test_predictions.append(predicted_labels.cpu())
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)


    def process_epoch_end(self, predictions, labels, phase):
        preds = np.concatenate([t.numpy() for t in predictions])
        lbls = np.concatenate([t.numpy() for t in labels])
        ranking_average_precision = MultilabelRankingAveragePrecision(num_labels=config.num_labels)

        f1_macro = f1_score(preds, lbls, average='macro')

        ranking_avg_precision = ranking_average_precision(torch.tensor(preds), torch.tensor(lbls))

        self.log(f"{phase} multilabel_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
        self.log(f"{phase} ranking_avg_precision", ranking_avg_precision, on_epoch=True, prog_bar=True)

        for i in range(config.num_labels):
            label_score = f1_score(lbls[:, i], preds[:, i])
            cm = confusion_matrix(lbls[:, i], np.round(preds[:, i]))
            print(f"Confusion Matrix for Label {i + 1}:\n{cm}\n")
            self.log(f"{i} {phase} f1", label_score, on_epoch=True, prog_bar=True)

        predictions.clear()
        labels.clear()

    def on_train_epoch_end(self):
        self.process_epoch_end(self.train_predictions, self.train_labels, "train")

    def on_validation_epoch_end(self):
        self.process_epoch_end(self.val_predictions, self.val_labels, "val")

    def on_test_epoch_end(self):
        self.process_epoch_end(self.test_predictions, self.test_labels, "test")


class BEiT(pl.LightningModule):
    def __init__(self):
        super(BEiT, self).__init__()

        self.vit = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k',
                                                              num_labels=config.num_labels,
                                                              problem_type="multi_label_classification",
                                                              id2label=config.id2label,
                                                              label2id=config.label2id,
                                                              ignore_mismatched_sizes=True)

        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []
        self.test_predictions = []
        self.test_labels = []
        self.epoch_train_loss = 0.0
        self.epoch_val_loss = 0.0
        self.train_steps = 0
        self.val_steps = 0

        self.train_ds, self.val_ds, self.test_ds = utils.load_data(dataset=config.dataset)


        for param in self.vit.parameters():
            param.requires_grad = False

        for param in self.vit.classifier.parameters():
            param.requires_grad = True


    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = config.criterion(logits, labels.float())

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        correct = (predicted_labels == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.train_labels.append(labels.cpu())
        self.train_predictions.append(predicted_labels.cpu())
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.val_labels.append(labels.cpu())
        self.val_predictions.append(predicted_labels.cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.test_labels.append(labels.cpu())
        self.test_predictions.append(predicted_labels.cpu())
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)


    def process_epoch_end(self, predictions, labels, phase):
        preds = np.concatenate([t.numpy() for t in predictions])
        lbls = np.concatenate([t.numpy() for t in labels])
        ranking_average_precision = MultilabelRankingAveragePrecision(num_labels=config.num_labels)

        f1_macro = f1_score(preds, lbls, average='macro')

        ranking_avg_precision = ranking_average_precision(torch.tensor(preds), torch.tensor(lbls))

        self.log(f"{phase} multilabel_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
        self.log(f"{phase} ranking_avg_precision", ranking_avg_precision, on_epoch=True, prog_bar=True)

        for i in range(config.num_labels):
            label_score = f1_score(lbls[:, i], preds[:, i])
            cm = confusion_matrix(lbls[:, i], np.round(preds[:, i]))
            print(f"Confusion Matrix for Label {i + 1}:\n{cm}\n")
            self.log(f"{i} {phase} f1", label_score, on_epoch=True, prog_bar=True)

        predictions.clear()
        labels.clear()

    def on_train_epoch_end(self):
        self.process_epoch_end(self.train_predictions, self.train_labels, "train")

    def on_validation_epoch_end(self):
        self.process_epoch_end(self.val_predictions, self.val_labels, "val")

    def on_test_epoch_end(self):
        self.process_epoch_end(self.test_predictions, self.test_labels, "test")


class LeViT(pl.LightningModule):
    def __init__(self):
        super(LeViT, self).__init__()

        self.vit = LevitForImageClassification.from_pretrained('facebook/levit-128S',
                                                              num_labels=config.num_labels,
                                                              problem_type="multi_label_classification",
                                                              id2label=config.id2label,
                                                              label2id=config.label2id,
                                                              ignore_mismatched_sizes=True)

        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []
        self.test_predictions = []
        self.test_labels = []
        self.epoch_train_loss = 0.0
        self.epoch_val_loss = 0.0
        self.train_steps = 0
        self.val_steps = 0

        self.train_ds, self.val_ds, self.test_ds = utils.load_data(dataset=config.dataset)


        for param in self.vit.parameters():
            param.requires_grad = False

        for param in self.vit.classifier.parameters():
            param.requires_grad = True


    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = config.criterion(logits, labels.float())

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        correct = (predicted_labels == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.train_labels.append(labels.cpu())
        self.train_predictions.append(predicted_labels.cpu())
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.val_labels.append(labels.cpu())
        self.val_predictions.append(predicted_labels.cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.test_labels.append(labels.cpu())
        self.test_predictions.append(predicted_labels.cpu())
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)


    def process_epoch_end(self, predictions, labels, phase):
        preds = np.concatenate([t.numpy() for t in predictions])
        lbls = np.concatenate([t.numpy() for t in labels])
        ranking_average_precision = MultilabelRankingAveragePrecision(num_labels=config.num_labels)

        f1_macro = f1_score(preds, lbls, average='macro')

        ranking_avg_precision = ranking_average_precision(torch.tensor(preds), torch.tensor(lbls))

        self.log(f"{phase} multilabel_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
        self.log(f"{phase} ranking_avg_precision", ranking_avg_precision, on_epoch=True, prog_bar=True)

        for i in range(config.num_labels):
            label_score = f1_score(lbls[:, i], preds[:, i])
            cm = confusion_matrix(lbls[:, i], np.round(preds[:, i]))
            print(f"Confusion Matrix for Label {i + 1}:\n{cm}\n")
            self.log(f"{i} {phase} f1", label_score, on_epoch=True, prog_bar=True)

        predictions.clear()
        labels.clear()

    def on_train_epoch_end(self):
        self.process_epoch_end(self.train_predictions, self.train_labels, "train")

    def on_validation_epoch_end(self):
        self.process_epoch_end(self.val_predictions, self.val_labels, "val")

    def on_test_epoch_end(self):
        self.process_epoch_end(self.test_predictions, self.test_labels, "test")

class DeiT(pl.LightningModule):
    def __init__(self):
        super(DeiT, self).__init__()

        self.vit = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                              num_labels=config.num_labels,
                                                              problem_type="multi_label_classification",
                                                              id2label=config.id2label,
                                                              label2id=config.label2id,
                                                              ignore_mismatched_sizes=True)

        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []
        self.test_predictions = []
        self.test_labels = []
        self.epoch_train_loss = 0.0
        self.epoch_val_loss = 0.0
        self.train_steps = 0
        self.val_steps = 0

        self.train_ds, self.val_ds, self.test_ds = utils.load_data(dataset=config.dataset)


        for param in self.vit.parameters():
            param.requires_grad = False

        for param in self.vit.classifier.parameters():
            param.requires_grad = True


    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = config.criterion(logits, labels.float())

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        correct = (predicted_labels == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.train_labels.append(labels.cpu())
        self.train_predictions.append(predicted_labels.cpu())
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.val_labels.append(labels.cpu())
        self.val_predictions.append(predicted_labels.cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.test_labels.append(labels.cpu())
        self.test_predictions.append(predicted_labels.cpu())
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)


    def process_epoch_end(self, predictions, labels, phase):
        preds = np.concatenate([t.numpy() for t in predictions])
        lbls = np.concatenate([t.numpy() for t in labels])
        ranking_average_precision = MultilabelRankingAveragePrecision(num_labels=config.num_labels)

        f1_macro = f1_score(preds, lbls, average='macro')

        ranking_avg_precision = ranking_average_precision(torch.tensor(preds), torch.tensor(lbls))

        self.log(f"{phase} multilabel_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
        self.log(f"{phase} ranking_avg_precision", ranking_avg_precision, on_epoch=True, prog_bar=True)

        for i in range(config.num_labels):
            label_score = f1_score(lbls[:, i], preds[:, i])
            cm = confusion_matrix(lbls[:, i], np.round(preds[:, i]))
            print(f"Confusion Matrix for Label {i + 1}:\n{cm}\n")
            self.log(f"{i} {phase} f1", label_score, on_epoch=True, prog_bar=True)

        predictions.clear()
        labels.clear()

    def on_train_epoch_end(self):
        self.process_epoch_end(self.train_predictions, self.train_labels, "train")

    def on_validation_epoch_end(self):
        self.process_epoch_end(self.val_predictions, self.val_labels, "val")

    def on_test_epoch_end(self):
        self.process_epoch_end(self.test_predictions, self.test_labels, "test")

class ImageGPT(pl.LightningModule):
    def __init__(self):
        super(ImageGPT, self).__init__()

        self.vit = ImageGPTForImageClassification.from_pretrained("openai/imagegpt-small",
                                                              num_labels=config.num_labels,
                                                              problem_type="multi_label_classification",
                                                              id2label=config.id2label,
                                                              label2id=config.label2id,
                                                              ignore_mismatched_sizes=True)

        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []
        self.test_predictions = []
        self.test_labels = []
        self.epoch_train_loss = 0.0
        self.epoch_val_loss = 0.0
        self.train_steps = 0
        self.val_steps = 0

        self.train_ds, self.val_ds, self.test_ds = utils.load_data(dataset=config.dataset)


        for param in self.vit.parameters():
            param.requires_grad = False

        for param in self.vit.score.parameters():
            param.requires_grad = True


    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = config.criterion(logits, labels.float())

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        correct = (predicted_labels == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.train_labels.append(labels.cpu())
        self.train_predictions.append(predicted_labels.cpu())
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.val_labels.append(labels.cpu())
        self.val_predictions.append(predicted_labels.cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.test_labels.append(labels.cpu())
        self.test_predictions.append(predicted_labels.cpu())
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def process_epoch_end(self, predictions, labels, phase):
        preds = np.concatenate([t.numpy() for t in predictions])
        lbls = np.concatenate([t.numpy() for t in labels])
        ranking_average_precision = MultilabelRankingAveragePrecision(num_labels=config.num_labels)

        f1_macro = f1_score(preds, lbls, average='macro')

        ranking_avg_precision = ranking_average_precision(torch.tensor(preds), torch.tensor(lbls))

        self.log(f"{phase} multilabel_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
        self.log(f"{phase} ranking_avg_precision", ranking_avg_precision, on_epoch=True, prog_bar=True)

        for i in range(config.num_labels):
            label_score = f1_score(lbls[:, i], preds[:, i])
            cm = confusion_matrix(lbls[:, i], np.round(preds[:, i]))
            print(f"Confusion Matrix for Label {i + 1}:\n{cm}\n")
            self.log(f"{i} {phase} f1", label_score, on_epoch=True, prog_bar=True)

        predictions.clear()
        labels.clear()

    def on_train_epoch_end(self):
        self.process_epoch_end(self.train_predictions, self.train_labels, "train")

    def on_validation_epoch_end(self):
        self.process_epoch_end(self.val_predictions, self.val_labels, "val")

    def on_test_epoch_end(self):
        self.process_epoch_end(self.test_predictions, self.test_labels, "test")

class ResNet50(pl.LightningModule):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.vit = ResNetForImageClassification.from_pretrained("microsoft/resnet-50",
                                                              num_labels=config.num_labels,
                                                              problem_type="multi_label_classification",
                                                              id2label=config.id2label,
                                                              label2id=config.label2id,
                                                              ignore_mismatched_sizes=True)

        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []
        self.test_predictions = []
        self.test_labels = []
        self.epoch_train_loss = 0.0
        self.epoch_val_loss = 0.0
        self.train_steps = 0
        self.val_steps = 0

        self.train_ds, self.val_ds, self.test_ds = utils.load_data(dataset=config.dataset)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = config.criterion(logits, labels.float())

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        correct = (predicted_labels == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.train_labels.append(labels.cpu())
        self.train_predictions.append(predicted_labels.cpu())
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.val_labels.append(labels.cpu())
        self.val_predictions.append(predicted_labels.cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.test_labels.append(labels.cpu())
        self.test_predictions.append(predicted_labels.cpu())
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)


    def process_epoch_end(self, predictions, labels, phase):
        preds = np.concatenate([t.numpy() for t in predictions])
        lbls = np.concatenate([t.numpy() for t in labels])
        ranking_average_precision = MultilabelRankingAveragePrecision(num_labels=config.num_labels)

        f1_macro = f1_score(preds, lbls, average='macro')

        ranking_avg_precision = ranking_average_precision(torch.tensor(preds), torch.tensor(lbls))

        self.log(f"{phase} multilabel_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
        self.log(f"{phase} ranking_avg_precision", ranking_avg_precision, on_epoch=True, prog_bar=True)

        for i in range(config.num_labels):
            label_score = f1_score(lbls[:, i], preds[:, i])
            cm = confusion_matrix(lbls[:, i], np.round(preds[:, i]))
            print(f"Confusion Matrix for Label {i + 1}:\n{cm}\n")
            self.log(f"{i} {phase} f1", label_score, on_epoch=True, prog_bar=True)

        predictions.clear()
        labels.clear()

    def on_train_epoch_end(self):
        self.process_epoch_end(self.train_predictions, self.train_labels, "train")

    def on_validation_epoch_end(self):
        self.process_epoch_end(self.val_predictions, self.val_labels, "val")

    def on_test_epoch_end(self):
        self.process_epoch_end(self.test_predictions, self.test_labels, "test")


class PASOK(pl.LightningModule):
    def __init__(self):
        super(PASOK, self).__init__()

        self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224",
                                                              num_labels=config.num_labels,
                                                              problem_type="multi_label_classification",
                                                              id2label=config.id2label,
                                                              label2id=config.label2id,
                                                              ignore_mismatched_sizes=True)

        self.train_labels = []
        self.train_predictions = []
        self.val_labels = []
        self.val_predictions = []
        self.test_predictions = []
        self.test_labels = []
        self.epoch_train_loss = 0.0
        self.epoch_val_loss = 0.0
        self.train_steps = 0
        self.val_steps = 0

        self.train_ds, self.val_ds, self.test_ds = utils.load_data(dataset=config.dataset)

        for param in self.vit.parameters():
            param.requires_grad = False

        for param in self.vit.classifier.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = config.criterion(logits, labels.float())

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()

        correct = (predicted_labels == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.train_labels.append(labels.cpu())
        self.train_predictions.append(predicted_labels.cpu())
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.val_labels.append(labels.cpu())
        self.val_predictions.append(predicted_labels.cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, labels, predicted_labels = self.common_step(batch, batch_idx)
        self.test_labels.append(labels.cpu())
        self.test_predictions.append(predicted_labels.cpu())
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, collate_fn=utils.collate_fn, batch_size=config.batch_size,
                          num_workers=4, persistent_workers=True, pin_memory=True)


    def process_epoch_end(self, predictions, labels, phase):
        preds = np.concatenate([t.numpy() for t in predictions])
        lbls = np.concatenate([t.numpy() for t in labels])
        ranking_average_precision = MultilabelRankingAveragePrecision(num_labels=config.num_labels)

        f1_macro = f1_score(preds, lbls, average='macro')

        ranking_avg_precision = ranking_average_precision(torch.tensor(preds), torch.tensor(lbls))

        self.log(f"{phase} multilabel_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
        self.log(f"{phase} ranking_avg_precision", ranking_avg_precision, on_epoch=True, prog_bar=True)

        for i in range(config.num_labels):
            label_score = f1_score(lbls[:, i], preds[:, i])
            cm = confusion_matrix(lbls[:, i], np.round(preds[:, i]))
            print(f"Confusion Matrix for Label {i + 1}:\n{cm}\n")
            self.log(f"{i} {phase} f1", label_score, on_epoch=True, prog_bar=True)

        predictions.clear()
        labels.clear()

    def on_train_epoch_end(self):
        self.process_epoch_end(self.train_predictions, self.train_labels, "train")

    def on_validation_epoch_end(self):
        self.process_epoch_end(self.val_predictions, self.val_labels, "val")

    def on_test_epoch_end(self):
        self.process_epoch_end(self.test_predictions, self.test_labels, "test")
