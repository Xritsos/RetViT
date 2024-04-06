# Test file
from matplotlib import pyplot as plt
from datasets import load_dataset
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics.classification import MultilabelF1Score
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

from source.train.dataloader import CustomImageDataset

# from model import ViTLightningModule

import pytorch_lightning as pl
from transformers import AutoModelForImageClassification, AdamW
import torch.nn as nn

class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=10):
        super(ViTLightningModule, self).__init__()
        self.vit = AutoModelForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224',
                                                              num_labels=8,
                                                              problem_type="multi_label_classification",
                                                              id2label=id2label,
                                                              label2id=label2id,
                                                              ignore_mismatched_sizes=True)
        
    # add this above for the multilabel case, plus more params values for multilabel                                  
    #  problem_type="multi_label_classification",

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
        
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels.float())
        # predictions = logits.argmax(-1) # for cross entropy
        predictions = logits
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]
        f1 = MultilabelF1Score(num_labels=8, average='weighted').to(device)
        f1 = f1(predictions, labels)

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
        return torch.optim.AdamW(self.parameters(), lr=5e-5)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

    def test_dataloader(self):
        return test_dataloader


# load_data
path_train = './data/train.csv'
path_val = './data/val.csv'
path_test = './data/test.csv'

root_train = './data/train/'
root_test = './data/test/'
root_val = './data/val/'

def train_transforms(examples):
    # examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    transformed = _train_transforms(examples) 
    return transformed

def val_transforms(examples):
    # examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    transformed = _val_transforms(examples)
    return transformed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_ds = CustomImageDataset(csv_file=path_train, root_dir=root_train, transform=train_transforms)
test_ds = CustomImageDataset(csv_file=path_test, root_dir=root_test, transform=val_transforms)
val_ds = CustomImageDataset(csv_file=path_val, root_dir=root_val, transform=val_transforms)

# integer to label mapping
# id2label = {id:label for id, label in enumerate(train_ds.data_frame['N'])}
id2label = {0: "N", 1: "D", 2: "G", 3: "C", 4: "A", 5: "H", 6: "M", 7: "O"}
# id2label = {0: "0", 1: "1"}
label2id = {label:id for id, label in id2label.items()}

# prepare images for inference
processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
image_mean = processor.image_mean
image_std = processor.image_std
# size = processor.size["height"]

# data augmentation - normalization
normalize = Normalize(mean=image_mean, std=image_std)

if "height" in processor.size:
    size = (processor.size["height"], processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in processor.size:
    size = processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = processor.size.get("longest_edge")

_train_transforms = Compose([RandomResizedCrop(crop_size),
                            RandomHorizontalFlip(), 
                            ToTensor(), 
                            normalize])

_val_transforms = Compose([Resize(size), 
                        CenterCrop(crop_size), 
                        ToTensor(), 
                        normalize])


# train_ds.set_transform(train_transforms)
# val_ds.set_transform(val_transforms)
# test_ds.set_transform(val_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = torch.permute(pixel_values, (0, 2, 1, 3))
   
    # labels = torch.tensor([example["label"] for example in examples])
    labels = torch.stack([example["label"] for example in examples])
    
    return {"pixel_values": pixel_values, "labels": labels}

train_batch_size = 10
eval_batch_size = 10

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size,
                              num_workers=4)

val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size,
                            num_workers=4)

test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size,
                             num_workers=4)

print(f"train_dataloader type: {type(train_dataloader)}")

batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k, v.shape)
    

# assert batch['pixel_values'].shape == (train_batch_size, 3, 224, 224)
# assert batch['labels'].shape == (train_batch_size,)

# next(iter(train_dataloader))['pixel_values'].shape


# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)

model = ViTLightningModule()
trainer = Trainer(accelerator='gpu', max_epochs=3, 
                  callbacks=[EarlyStopping(monitor='validation_loss')])

trainer.fit(model)

print()
print("================ Testing... ======================")
trainer.test(ckpt_path='best')
print("==================================================")
