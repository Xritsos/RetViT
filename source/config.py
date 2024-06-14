# Configuration file for various experiments

import torch.nn as nn
import torch

dataset = 'ODIR'
# Experiment setup parameters
id2label = {0: "N", 1: "D", 2: "C", 3: "M"}
label2id = {label:id for id, label in id2label.items()}


num_labels = 4
class_weights = torch.tensor([1.03297754e-05, 1.27997310e-05, 7.08251781e-05, 8.47295075e-05], dtype=torch.float).to('cuda')
criterion = nn.BCEWithLogitsLoss()

# Training Parameters
learning_rate = 0.0005
weight_decay = 0.0
batch_size = 128

early_stopping_patience = 10
num_epochs = 400

model_processor = 'BEiT'  # 'SWIN' 'LeViT' 'BEiT' 'VIT' 'DeiT' ResNet
