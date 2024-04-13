import pandas as pd
import torch.nn as nn
from source.train.dataloader import CustomImageDataset
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)
from transformers import AutoImageProcessor
import torch


# prepare images for inference
processor = AutoImageProcessor.from_pretrained("microsoft/swin-large-patch4-window12-384")
image_mean = processor.image_mean
image_std = processor.image_std

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


def get_params(config_path):
    # Read the CSV file
    config = pd.read_csv(config_path)

    # Extract values from the DataFrame
    learning_rate = config.loc[0, 'learning_rate']
    weight_decay = config.loc[0, 'weight_decay']
    batch_size = int(config.loc[0, 'batch_size'])
    early_stopping_patience = int(config.loc[0, 'early_stopping_patience'])
    num_epochs = int(config.loc[0, 'num_epochs'])
    criterion = nn.BCEWithLogitsLoss()

    return learning_rate, weight_decay, batch_size, early_stopping_patience, num_epochs, criterion


def load_data():
    # load_data
    path_train = './data/train.csv'
    path_val = './data/val.csv'
    path_test = './data/test.csv'

    root_train = './data/train/'
    root_val = './data/val/'
    root_test = './data/test/'

    def train_transforms(examples):
        # examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        transformed = _train_transforms(examples)
        return transformed

    def val_transforms(examples):
        # examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        transformed = _val_transforms(examples)
        return transformed

    train_ds = CustomImageDataset(csv_file=path_train, root_dir=root_train, transform=train_transforms)
    val_ds = CustomImageDataset(csv_file=path_val, root_dir=root_val, transform=val_transforms)
    test_ds = CustomImageDataset(csv_file=path_test, root_dir=root_test, transform=val_transforms)

    _train_transforms = Compose([RandomResizedCrop(crop_size),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 normalize])

    _val_transforms = Compose([Resize(size),
                               CenterCrop(crop_size),
                               ToTensor(),
                               normalize])

    return train_ds, val_ds, test_ds


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = torch.permute(pixel_values, (0, 2, 1, 3))

    # labels = torch.tensor([example["label"] for example in examples])
    labels = torch.stack([example["label"] for example in examples])

    return {"pixel_values": pixel_values, "labels": labels}