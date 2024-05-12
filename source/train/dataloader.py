import os
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import (Compose, Normalize, 
                                    RandomHorizontalFlip, 
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_names = ["N", "D", "C", "M", "O"]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 3])
        image = Image.open(img_name)
        label = []
        
        for name in self.label_names:
            if self.data_frame.at[idx, name] == 1:
                label.append(1)
            else:
                label.append(0)
                
        label = torch.tensor(label)
                
        # label = self.data_frame.at[idx, 'N']
        
        # Apply your preprocessing pipeline
        if self.transform:
            image = self.transform(image)
        
        image_tensor = torch.tensor(np.array(image))
        image_tensor = torch.permute(image_tensor, (2, 0, 1))
        
        # this return dictionary is returned in the collate_fn function
        # this might be useless and should be removed ?
        return {"img": image, "label": label, "pixel_values": image_tensor}
