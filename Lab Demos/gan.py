# Multi-GPU DCGAN for OASIS Brain MRI Generation
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
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ================================
# GPU AND DISTRIBUTED SETUP
# ================================

def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def get_device_config():
    """Configure device settings for multi-GPU setup"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA available with {device_count} GPU(s)")
        return 'cuda', device_count
    elif torch.backends.mps.is_available():
        print("MPS available (single device)")
        return 'mps', 1
    else:
        print("Using CPU")
        return 'cpu', 1

# Device configuration
device_type, num_gpus = get_device_config()
use_distributed = num_gpus > 1 and device_type == 'cuda'

print(f"Device type: {device_type}")
print(f"Number of GPUs: {num_gpus}")
print(f"Using distributed training: {use_distributed}")

# ================================
# DATASET CONFIGURATION
# ================================

# Dataset paths
OASIS_PATH = "/home/groups/comp3710/"
TRAIN_IMAGES_PATH = os.path.join(OASIS_PATH, "keras_png_slices_train")
TRAIN_MASKS_PATH = os.path.join(OASIS_PATH, "keras_png_slices_seg_train")
TEST_IMAGES_PATH = os.path.join(OASIS_PATH, "keras_png_slices_test")
TEST_MASKS_PATH = os.path.join(OASIS_PATH, "keras_png_slices_seg_test")

class OASISDataset(Dataset):
    """Custom dataset for OASIS brain MRI images - Multi-GPU optimized"""
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
        image = Image.open(self.image_files[idx]).convert('L') # Convert to grayscale
        image = image.resize(self.target_size)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        if self.mask_files is not None:
            mask = Image.open(self.mask_files[idx]).convert('L')
            mask = mask.resize(self.target_size)
            mask = transforms.ToTensor()(mask)
            return image, mask
        
        return image

# ================================
# MULTI-GPU OPTIMIZED MODELS
# ================================

class Generator(nn.Module):
    """Generator network for DCGAN - Multi-GPU optimized"""
    def __init__(self, nz=100, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            # State size: (ngf//2) x 64 x 64
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # State size: (nc) x 128 x 128
        )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    """Discriminator network for DCGAN - Multi-GPU optimized"""
    def __init__(self, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf//2) x 64 x 64
            nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def weights_init(m):
    """Initialize weights for DCGAN"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ================================
# MULTI-GPU TRAINING FUNCTIONS
# ================================

def create_data_loader(dataset, batch_size, rank=0, world_size=1, use_distributed=False):
    """Create data loader with optional distributed sampling"""
    if use_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                          num_workers=2, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    return loader

def train_gan_distributed(rank, world_size):
    """Main training function for distributed training"""
    print(f"Running DDP on rank {rank}")
    setup_distributed(rank, world_size)
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Data preparation
    transform_gan = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset_gan = OASISDataset(TRAIN_IMAGES_PATH, transform=transform_gan, target_size=(128, 128))
    train_loader_gan = create_data_loader(train_dataset_gan, batch_size=32, 
                                        rank=rank, world_size=world_size, 
                                        use_distributed=True)
    
    # Model initialization
    nz = 100
    netG = Generator(nz=nz).to(device)
    netD = Discriminator().to(device)
    
    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Wrap models in DDP
    netG = DDP(netG, device_ids=[rank])
    netD = DDP(netD, device_ids=[rank])
    
    # Optimizers
    lr = 0.0002
    beta1 = 0.5
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training parameters
    gan_epochs = 200  # Reduced for multi-GPU demo
    print_every = 20
    
    # Fixed noise for monitoring progress (only on rank 0)
    if rank == 0:
        fixed_noise = torch.randn(16, nz, 1, 1, device=device)
    
    if rank == 0:
        print(f"Training GAN on {len(train_dataset_gan)} images...")
        print(f"Generator parameters: {sum(p.numel() for p in netG.parameters())}")
        print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters())}")
    
    # Training loop
    G_losses = []
    D_losses = []
    
    for epoch in range(gan_epochs):
        epoch_G_loss = 0.0
        epoch_D_loss = 0.0
        
        # Set epoch for distributed sampler
        train_loader_gan.sampler.set_epoch(epoch)
        
        for i, data in enumerate(train_loader_gan):
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            
            real_data = data.to(device, non_blocking=True)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
            
            output = netD(real_data)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Train with fake batch
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.0)
            
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            label.fill_(1.0)
            
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            epoch_G_loss += errG.item()
            epoch_D_loss += errD.item()
        
        # Average losses
        G_losses.append(epoch_G_loss / len(train_loader_gan))
        D_losses.append(epoch_D_loss / len(train_loader_gan))
        
        # Only print and save from rank 0
        if rank == 0 and (epoch + 1) % print_every == 0:
            print(f'[{epoch+1}/{gan_epochs}] Loss_D: {D_losses[-1]:.4f} Loss_G: {G_losses[-1]:.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
            
            # Generate sample images
            with torch.no_grad():
                fake_samples = netG(fixed_noise)
                
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for idx in range(16):
                    img = fake_samples[idx].cpu().detach().squeeze().numpy()
                    axes[idx//4, idx%4].imshow(img, cmap='gray')
                    axes[idx//4, idx%4].axis('off')
                plt.suptitle(f'Generated Brain Images - Epoch {epoch+1}')
                plt.tight_layout()
                plt.savefig(f'gan_distributed_epoch_{epoch+1}.png')
                plt.close()
    
    # Save final results (only from rank 0)
    if rank == 0:
        save_final_results(netG, G_losses, D_losses, device, nz, train_loader_gan)
    
    cleanup_distributed()

def train_gan_single_gpu():
    """Training function for single GPU or CPU"""
    if device_type == 'cuda':
        device = torch.device('cuda:0')
    elif device_type == 'mps':
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Training on single device: {device}")
    
    # Data preparation
    transform_gan = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset_gan = OASISDataset(TRAIN_IMAGES_PATH, transform=transform_gan, target_size=(128, 128))
    train_loader_gan = create_data_loader(train_dataset_gan, batch_size=64, use_distributed=False)  # Larger batch for single GPU
    
    # Model initialization
    nz = 100
    netG = Generator(nz=nz).to(device)
    netD = Discriminator().to(device)
    
    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Use DataParallel if multiple GPUs available but not using distributed
    if device_type == 'cuda' and num_gpus > 1:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        print(f"Using DataParallel with {num_gpus} GPUs")
    
    # Continue with original training logic...
    # [Rest of training code similar to original but with device optimization]
    
    # Optimizers
    lr = 0.0002
    beta1 = 0.5
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training parameters
    gan_epochs = 500  # More epochs for single GPU
    print_every = 50
    
    fixed_noise = torch.randn(16, nz, 1, 1, device=device)
    
    print(f"Training GAN on {len(train_dataset_gan)} images...")
    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters())}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters())}")
    
    # Training loop (similar to original but optimized)
    G_losses = []
    D_losses = []
    
    for epoch in range(gan_epochs):
        epoch_G_loss = 0.0
        epoch_D_loss = 0.0
        
        for i, data in enumerate(train_loader_gan):
            # Update Discriminator
            netD.zero_grad()
            
            real_data = data.to(device, non_blocking=True)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
            
            output = netD(real_data)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.0)
            
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # Update Generator
            netG.zero_grad()
            label.fill_(1.0)
            
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            epoch_G_loss += errG.item()
            epoch_D_loss += errD.item()
        
        G_losses.append(epoch_G_loss / len(train_loader_gan))
        D_losses.append(epoch_D_loss / len(train_loader_gan))
        
        if (epoch + 1) % print_every == 0:
            print(f'[{epoch+1}/{gan_epochs}] Loss_D: {D_losses[-1]:.4f} Loss_G: {G_losses[-1]:.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
            
            with torch.no_grad():
                fake_samples = netG(fixed_noise)
                
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                for idx in range(16):
                    img = fake_samples[idx].cpu().detach().squeeze().numpy()
                    axes[idx//4, idx%4].imshow(img, cmap='gray')
                    axes[idx//4, idx%4].axis('off')
                plt.suptitle(f'Generated Brain Images - Epoch {epoch+1}')
                plt.tight_layout()
                plt.savefig(f'gan_single_gpu_epoch_{epoch+1}.png')
                plt.close()
    
    save_final_results(netG, G_losses, D_losses, device, nz, train_loader_gan)

def save_final_results(netG, G_losses, D_losses, device, nz, train_loader):
    """Save final training results and comparisons"""
    print("Saving final results...")
    
    # Plot training losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.title('GAN Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Generate final results
    netG.eval()
    with torch.no_grad():
        test_noise = torch.randn(64, nz, 1, 1, device=device)
        final_generated = netG(test_noise)
        
        plt.subplot(1, 2, 2)
        fig, axes = plt.subplots(8, 8, figsize=(16, 16))
        for idx in range(64):
            img = final_generated[idx].cpu().detach().squeeze().numpy()
            axes[idx//8, idx%8].imshow(img, cmap='gray')
            axes[idx//8, idx%8].axis('off')
        plt.suptitle('Final Generated Brain Images from Multi-GPU DCGAN')
        plt.tight_layout()
        plt.savefig('gan_multi_gpu_final_results.png')
        plt.show()
    
    print("Training completed and results saved!")

# ================================
# MAIN EXECUTION
# ================================

def main():
    """Main function to choose training mode"""
    print("Multi-GPU DCGAN for Brain MRI Generation")
    print("=" * 50)
    
    if not os.path.exists(TRAIN_IMAGES_PATH):
        print(f"ERROR: Training images path not found: {TRAIN_IMAGES_PATH}")
        return
    
    if use_distributed and num_gpus > 1:
        print(f"Using distributed training across {num_gpus} GPUs")
        mp.spawn(train_gan_distributed, 
                args=(num_gpus,), 
                nprocs=num_gpus, 
                join=True)
    else:
        print("Using single GPU/CPU training")
        train_gan_single_gpu()

if __name__ == "__main__":
    main()