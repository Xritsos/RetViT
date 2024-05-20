import pandas as pd
from source.train.dataloader import CustomImageDataset
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)
from torchvision.transforms import RandomRotation, ColorJitter, RandomAffine, RandomVerticalFlip

from transformers import AutoImageProcessor, BeitImageProcessor, LevitImageProcessor, DeiTModel
import torch
import matplotlib.pyplot as plt
import sys
import os
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def plot_results():
    base_path = 'logs/experiment_name/'

    subdirs = glob.glob(os.path.join(base_path, '*/'))

    if not subdirs:
        print("No subdirectories found in the experiment_name directory.")
        return

    latest_dir = max(subdirs, key=os.path.getctime)

    metrics_file = os.path.join(latest_dir, 'metrics.csv')

    if not os.path.isfile(metrics_file):
        print(f"No metrics.csv file found in the latest directory: {latest_dir}")
        return
    else:
        print(f"plotting results from {metrics_file}")

    # Read the metrics.csv file
    metrics_df = pd.read_csv(metrics_file)
    fig, axs = plt.subplots(1, config.num_labels, figsize=(16, 8))

    for i in range(config.num_labels):
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
if config.model_processor == 'SWIN':
    processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
elif config.model_processor == 'BEiT':
    processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
elif config.model_processor == 'LeViT':
    processor = LevitImageProcessor.from_pretrained("facebook/levit-128S")
elif config.model_processor == 'DeiT':
    processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
elif config.model_processor == 'ImageGPT':
    processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
elif config.model_processor == 'ResNet':
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
elif config.model_processor == 'VIT':
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
else:
    print('NO COMPATIBLE MODEL ADDED. Moving on with VIT.')
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

image_mean = processor.image_mean
image_std = processor.image_std

# data augmentation - normalization
normalize = Normalize(mean=image_mean, std=image_std)

if "height" in processor.size:
    size = (224, 224)  # Sos this is hardcoded in case of error use dynamic processor values
    # crop_size = size
    max_size = None
elif "shortest_edge" in processor.size:
    size = processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = processor.size.get("longest_edge")
size = (224, 224)

def load_data(dataset):
    # load_data

    if dataset == 'ODIR':
        path_train = 'data/ODIR/train.csv'
        path_val = 'data/ODIR/val.csv'
        path_test = 'data/ODIR/test.csv'

        root_train = 'data/ODIR/train/'
        root_val = 'data/ODIR/val/'
        root_test = 'data/ODIR/test/'
    elif dataset == 'RFMID':
        path_train = './data/RFMiD/RFMiD_Training_Labels_curated.csv'
        path_val = './data/RFMiD/RFMiD_Validation_Labels_curated.csv'
        path_test = './data/RFMiD/RFMiD_Testing_Labels_curated.csv'

        root_train = './data/RFMiD/images/'
        root_val = './data/RFMiD/images/'
        root_test = './data/RFMiD/images/'
    else:
        raise ValueError('Non-accepted dataset. Use ODIR or RFMID')

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
                               ToTensor(),
                               normalize])

    return train_ds, val_ds, test_ds


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = torch.permute(pixel_values, (0, 2, 1, 3))

    labels = torch.stack([example["label"] for example in examples])

    return {"pixel_values": pixel_values, "labels": labels}
