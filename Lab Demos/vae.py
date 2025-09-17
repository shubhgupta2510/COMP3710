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
    
    # Check for the number of images
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_files[idx]).convert('L')  # Convert to grayscale
        image = image.resize(self.target_size)
        
        # Apply transform if given, else, a float tensor in [0,1]
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

# Transform for VAE (normalise to [-1, 1] for Tanh output for the decoder)
transform_vae = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Transform for UNet (normalise to [0, 1])
transform_unet = transforms.Compose([
    transforms.ToTensor(),
])

print("Dataset class defined successfully!")

# Implementation of a VAE for MRI brain images from the OASIS dataset. 
# The VAE learns a latent representation that can be used to generate new brain images and visualise the manifold.

class VAE(nn.Module):
    """Variational Autoencoder for brain MRI images"""
    def __init__(self, input_dim=128*128, hidden_dim=512, latent_dim=64):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Latent space parameters
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x.view(-1, self.input_dim))
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterise(self, mu, logvar):
        """Reparameterisation trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent variable to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE loss function combining reconstruction and KL divergence"""
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x.view(-1, 128*128), x.view(-1, 128*128), reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss, recon_loss, kl_loss

print("VAE model defined successfully!")

# Load data for VAE
train_dataset_vae = OASISDataset(TRAIN_IMAGES_PATH, transform=transform_vae, target_size=(128, 128))
train_loader_vae = DataLoader(train_dataset_vae, batch_size=32, shuffle=True)

# Initialise VAE model
vae_model = VAE(latent_dim=64).to(device)
vae_optimiser = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

# Training parameters
vae_epochs = 50
print_every = 5

print(f"Training VAE on {len(train_dataset_vae)} images...")
print(f"Device: {device}")

# Training loop
vae_model.train()
train_losses = []

for epoch in range(vae_epochs):
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    
    for batch_idx, data in enumerate(train_loader_vae):
        data = data.to(device)
        
        vae_optimiser.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = vae_model(data)
        
        # Calculate loss
        loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar)
        
        # Backward pass
        loss.backward()
        vae_optimiser.step()
        
        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()
    
    # Average losses
    avg_loss = epoch_loss / len(train_loader_vae.dataset)
    avg_recon_loss = epoch_recon_loss / len(train_loader_vae.dataset)
    avg_kl_loss = epoch_kl_loss / len(train_loader_vae.dataset)
    
    train_losses.append(avg_loss)
    
    if (epoch + 1) % print_every == 0:
        print(f'Epoch [{epoch+1}/{vae_epochs}], Total Loss: {avg_loss:.4f}, '
              f'Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}')

print("VAE training completed!")

# VAE Visualisation and Manifold Exploration
import umap
from sklearn.decomposition import PCA

# Plot training loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('VAE Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('vae_training_loss.png')

# Generate reconstructions
vae_model.eval()
with torch.no_grad():
    # Get a batch of test images
    test_images = next(iter(train_loader_vae))[:8].to(device)
    reconstructions, mu, logvar = vae_model(test_images)
    
    # Plot original vs reconstructed images
    plt.subplot(1, 2, 2)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    for i in range(8):
        # Original images
        orig_img = test_images[i].cpu().squeeze()
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed images
        recon_img = reconstructions[i].cpu().view(128, 128)
        axes[1, i].imshow(recon_img, cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('vae_reconstructions.png')

# Generate new samples from the learned manifold
print("Generating new brain images from latent space...")
with torch.no_grad():
    # Sample from latent space
    z_sample = torch.randn(16, 64).to(device)
    generated_images = vae_model.decode(z_sample)
    
    # Plot generated images
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(16):
        img = generated_images[i].cpu().view(128, 128)
        axes[i//4, i%4].imshow(img, cmap='gray')
        axes[i//4, i%4].set_title(f'Generated {i+1}')
        axes[i//4, i%4].axis('off')
    
    plt.suptitle('Generated Brain Images from VAE Latent Space')
    plt.tight_layout()
    plt.show()
    plt.savefig('vae_generated_images.png')

print("VAE visualisation completed!")