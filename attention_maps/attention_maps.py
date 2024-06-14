import os.path

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import BeitImageProcessor
from models.models import BEiT


def load_model(checkpoint_path):
    model = BEiT.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, image_size=224):
    image = Image.open(image_path)
    transforms = Compose([
        Resize((image_size, image_size)),
        CenterCrop((image_size, image_size)),
        ToTensor(),
        normalize
    ])
    image_tensor = transforms(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)


def infer_attention_maps(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        attentions = outputs[1]
    return attentions


def overlay_heatmap(image, heatmap, alpha=0.7, cmap='hot'):
    resized_heatmap = np.array(Image.fromarray(heatmap).resize((image.shape[1], image.shape[0])))

    heatmap_normalized = (resized_heatmap - resized_heatmap.min()) / (resized_heatmap.max() - resized_heatmap.min())

    heatmap_colored = plt.get_cmap(cmap)(heatmap_normalized)

    # blend the heatmap with the original image
    overlaid_image = (alpha * heatmap_colored[:, :, :3] + (1 - alpha) * image / 255.0)

    # clip values to range [0, 1]
    overlaid_image = np.clip(overlaid_image, 0, 1)
    plt.figure(figsize=(15, 7))

    # plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # plot the overlaid image
    plt.subplot(1, 2, 2)
    plt.imshow(overlaid_image)
    plt.title('Attention Map')
    plt.axis('off')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
    plt.show()


# Load the image processor and the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
image_mean = processor.image_mean
image_std = processor.image_std
normalize = Normalize(mean=image_mean, std=image_std)


def visualize_attention(image_path, checkpoint_path):
    model = load_model(checkpoint_path)
    image_tensor = preprocess_image(image_path)
    attentions = infer_attention_maps(model, image_tensor)

    last_attention = attentions[-1]
    cls_attn_map = last_attention[:, :, 0, 1:].mean(dim=1).view(14, 14).cpu().numpy()
    cls_attn_map_normalized = (cls_attn_map - cls_attn_map.min()) / (cls_attn_map.max() - cls_attn_map.min())

    original_image = Image.open(image_path)
    original_image_np = np.asarray(original_image)

    overlay_heatmap(original_image_np, cls_attn_map_normalized)


train_folder = r"/home/t/tdrosog/RetViT/data/ODIR/train/"
val_folder = r"/home/t/tdrosog/RetViT/data/ODIR/val/"
test_folder = r"/home/t/tdrosog/RetViT/data/ODIR/test/"

files = ['16_right.jpg', '1164_left.jpg', '1164_right.jpg', '2100_left.jpg', '1543_left.jpg', '1543_right.jpg',
         '1_left.jpg', '1_right.jpg', '16_left.jpg', '2164_left.jpg', '2164_right.jpg', '2100_right.jpg']
checkpoint_path = '/home/t/tdrosog/RetViT/logs/experiment_name/version_6/checkpoints/epoch=24-step=4325.ckpt'

folders = [train_folder, val_folder, test_folder]

for file_path in files:
    for folder in folders:
        try:
            image_path = os.path.join(folder, file_path)
            visualize_attention(image_path, checkpoint_path)
            break
        except FileNotFoundError:
            continue
    else:
        print(f'File {file_path} not found in any folder.')
