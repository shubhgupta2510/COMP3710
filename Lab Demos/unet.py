# Data loading and preprocessing for OASIS dataset
import os
import glob
from PIL import Image
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    print("Using MPS")
elif torch.cuda.is_available():
    print("Using CUDA")
else:
    print("Using CPU")

# Dataset paths
OASIS_PATH = "/home/groups/comp3710/"
TRAIN_IMAGES_PATH = os.path.join(OASIS_PATH, "keras_png_slices_train")
TRAIN_MASKS_PATH = os.path.join(OASIS_PATH, "keras_png_slices_seg_train")
TEST_IMAGES_PATH = os.path.join(OASIS_PATH, "keras_png_slices_test")
TEST_MASKS_PATH = os.path.join(OASIS_PATH, "keras_png_slices_seg_test")

print("Data paths configured successfully!")
print(f"Train images path exists: {os.path.exists(TRAIN_IMAGES_PATH)}")
print(f"Train masks path exists: {os.path.exists(TRAIN_MASKS_PATH)}")

class OASISDataset(Dataset):
    """Custom dataset for OASIS brain MRI images"""
    def __init__(self, images_path, masks_path=None, transform=None, target_size=(128, 128)):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.target_size = target_size
        
        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(images_path, "*.png")))
        
        if masks_path:
            self.mask_files = sorted(glob.glob(os.path.join(masks_path, "*.png")))
            assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"
        else:
            self.mask_files = None
            
        print(f"Found {len(self.image_files)} images")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_files[idx]).convert('L')  # Convert to grayscale
        image = image.resize(self.target_size)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        if self.mask_files is not None:
            # Load mask for segmentation tasks
            mask = Image.open(self.mask_files[idx]).convert('L')
            mask = mask.resize(self.target_size)
            mask = transforms.ToTensor()(mask)
            return image, mask
        
        return image

# Transform for VAE (normalise to [-1, 1])
transform_vae = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Transform for UNet (normalise to [0, 1])
transform_unet = transforms.Compose([
    transforms.ToTensor(),
])

print("Dataset class defined successfully!")

# Implementation of UNet architecture for brain MRI segmentation. 
# The model uses skip connections to preserve spatial information and achieve high segmentation accuracy (>0.9 DSC).

class DoubleConv(nn.Module):
    """Double convolution block used in UNet"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """UNet architecture for image segmentation"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNet (Encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up part of UNet (Decoder)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse the list
        
        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsampling
            skip_connection = skip_connections[idx//2]
            
            # Handle size mismatch by cropping or padding
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # Double conv
        
        return torch.sigmoid(self.final_conv(x))

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice Similarity Coefficient"""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for training"""
    return 1 - dice_coefficient(pred, target, smooth)

print("UNet model defined successfully!")

# Load data for UNet segmentation
train_dataset_unet = OASISDataset(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, 
                                  transform=transform_unet, target_size=(128, 128))
train_loader_unet = DataLoader(train_dataset_unet, batch_size=16, shuffle=True)

test_dataset_unet = OASISDataset(TEST_IMAGES_PATH, TEST_MASKS_PATH, 
                                 transform=transform_unet, target_size=(128, 128))
test_loader_unet = DataLoader(test_dataset_unet, batch_size=16, shuffle=False)

# Initialize UNet model
unet_model = UNet(in_channels=1, out_channels=1).to(device)
unet_optimizer = torch.optim.Adam(unet_model.parameters(), lr=1e-3)

# Training parameters
unet_epochs = 30
print_every = 5

print(f"Training UNet on {len(train_dataset_unet)} image-mask pairs...")
print(f"Test set size: {len(test_dataset_unet)}")
print(f"Device: {device}")

# Training loop
unet_model.train()
train_losses_unet = []
dice_scores = []

for epoch in range(unet_epochs):
    epoch_loss = 0.0
    epoch_dice = 0.0
    
    for batch_idx, (images, masks) in enumerate(train_loader_unet):
        images, masks = images.to(device), masks.to(device)
        
        unet_optimizer.zero_grad()
        
        # Forward pass
        predictions = unet_model(images)
        
        # Calculate loss (combination of BCE and Dice loss)
        bce_loss = F.binary_cross_entropy(predictions, masks)
        d_loss = dice_loss(predictions, masks)
        loss = bce_loss + d_loss
        
        # Calculate Dice coefficient
        dice_score = dice_coefficient(predictions, masks)
        
        # Backward pass
        loss.backward()
        unet_optimizer.step()
        
        epoch_loss += loss.item()
        epoch_dice += dice_score.item()
    
    # Average losses
    avg_loss = epoch_loss / len(train_loader_unet)
    avg_dice = epoch_dice / len(train_loader_unet)
    
    train_losses_unet.append(avg_loss)
    dice_scores.append(avg_dice)
    
    if (epoch + 1) % print_every == 0:
        print(f'Epoch [{epoch+1}/{unet_epochs}], Loss: {avg_loss:.4f}, '
              f'Dice Score: {avg_dice:.4f}')

print("UNet training completed!")

# UNet Evaluation and Visualization
# Test the model
unet_model.eval()
test_dice_scores = []

print("Evaluating UNet on test set...")
with torch.no_grad():
    total_dice = 0.0
    num_batches = 0
    
    for images, masks in test_loader_unet:
        images, masks = images.to(device), masks.to(device)
        predictions = unet_model(images)
        
        # Calculate dice score for each image in batch
        for i in range(images.size(0)):
            dice_score = dice_coefficient(predictions[i], masks[i])
            test_dice_scores.append(dice_score.item())
        
        batch_dice = dice_coefficient(predictions, masks)
        total_dice += batch_dice.item()
        num_batches += 1
    
    avg_test_dice = total_dice / num_batches
    print(f"Average Test Dice Score: {avg_test_dice:.4f}")

# Plot training progress
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses_unet)
plt.title('UNet Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(dice_scores)
plt.title('UNet Training Dice Score')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.hist(test_dice_scores, bins=20, alpha=0.7)
plt.title('Test Dice Score Distribution')
plt.xlabel('Dice Score')
plt.ylabel('Frequency')
plt.axvline(x=0.9, color='r', linestyle='--', label='Target (0.9)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.savefig('unet_training_progress.png')

# Visualize segmentation results
print("Visualizing segmentation results...")
unet_model.eval()
with torch.no_grad():
    # Get a batch of test images
    test_images, test_masks = next(iter(test_loader_unet))
    test_images, test_masks = test_images.to(device), test_masks.to(device)
    predictions = unet_model(test_images)
    
    # Plot results
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    
    for i in range(6):
        # Original images
        orig_img = test_images[i].cpu().squeeze()
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Ground truth masks
        gt_mask = test_masks[i].cpu().squeeze()
        axes[1, i].imshow(gt_mask, cmap='gray')
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')
        
        # Predicted masks
        pred_mask = predictions[i].cpu().squeeze()
        axes[2, i].imshow(pred_mask, cmap='gray')
        dice_score = dice_coefficient(predictions[i], test_masks[i]).item()
        axes[2, i].set_title(f'Predicted (DSC: {dice_score:.3f})')
        axes[2, i].axis('off')
    
    plt.suptitle('UNet Segmentation Results')
    plt.tight_layout()
    plt.show()
    plt.savefig('unet_segmentation_results.png')

print("UNet evaluation completed!")