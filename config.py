# Configuration file for various experiments

import torch.nn as nn
import torch

dataset = 'ODIR'
# Experiment setup parameters
id2label = {0: "N", 1: "D", 2: "C", 3: "M"}
label2id = {label:id for id, label in id2label.items()}


num_labels = 4
# class_weights = torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.float).to('cuda')
criterion = nn.BCEWithLogitsLoss()

# Training Parameters
learning_rate = 0.0005
weight_decay = 0
batch_size = 16

early_stopping_patience = 10
num_epochs = 400

model_processor = 'VIT' # 'LeViT'  #'ImageGPT'  #  # 'DeiT'  #  # 'SWIN' # 'BEiT'
