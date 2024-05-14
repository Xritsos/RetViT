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
import matplotlib.pyplot as plt


def plot_results():
    metrics_df = pd.read_csv('logs/experiment_name/version_5/metrics.csv')

    fig, axs = plt.subplots(1, 4, figsize=(16, 8))

    for i in range(4):
        label_train_f1 = metrics_df.groupby('epoch')[f'{i} train f1'].mean()
        label_val_f1 = metrics_df.groupby('epoch')[f'{i} val f1'].mean()
        epochs = range(1, len(label_train_f1) + 1)


        axs[i].plot(epochs, label_train_f1, 'b', label=f'Mean Training F1 for label {i}')
        axs[i].plot(epochs, label_val_f1, 'r', label=f'Mean Validation F1 for label {i}')
        axs[i].set_title(f'Train - Val F1 for label {i}')
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('F1 Score')
        axs[i].legend()
        axs[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

# prepare images for inference
#processor = AutoImageProcessor.from_pretrained("microsoft/swin-large-patch4-window12-384")
processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
image_mean = processor.image_mean
image_std = processor.image_std

# data augmentation - normalization
normalize = Normalize(mean=image_mean, std=image_std)

if "height" in processor.size:
    size = (processor.size["height"], processor.size["width"])
    # crop_size = size
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

    # class_weights = torch.tensor([2.21, 3.74, 21.66, 21.76, 23.11, 32.56, 26.09, 7.16], dtype=torch.float).to('cuda')
    # weight = class_weights
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

    _train_transforms = Compose([Resize(size),
                                 RandomHorizontalFlip(),
                                 ToTensor(),
                                 normalize])

    _val_transforms = Compose([Resize(size),
                               # CenterCrop(crop_size),
                               ToTensor(),
                               normalize])

    return train_ds, val_ds, test_ds


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = torch.permute(pixel_values, (0, 2, 1, 3))

    # labels = torch.tensor([example["label"] for example in examples])
    labels = torch.stack([example["label"] for example in examples])

    return {"pixel_values": pixel_values, "labels": labels}
