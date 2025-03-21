import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# Image dimensions for 1024x1024 RGB images
IMG_HEIGHT, IMG_WIDTH = 1024, 1024
IMG_CHANNELS = 3

def load_images_from_folder(folder, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Load and preprocess images from a given folder.
    Images are resized to target_size, converted to numpy arrays,
    normalized to [0, 1] and rearranged to (channels, height, width).
    """
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.resize(target_size)
                img = np.array(img)
                # If image has an alpha channel, drop it.
                if img.shape[-1] == 4:
                    img = img[..., :3]
                img = img.astype('float32') / 255.0
                # Convert HWC to CHW format
                img = np.transpose(img, (2, 0, 1))
                images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(images)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: progressively reduce spatial dimensions
        self.encoder = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, 32, kernel_size=3, padding=1),  # 1024x1024 -> 1024x1024x32
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                              # 512x512x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),            # 512x512x64
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                              # 256x256x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # 256x256x128
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                              # 128x128x128

            nn.Conv2d(128, 256, kernel_size=3, padding=1),          # 128x128x256
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                              # 64x64x256

            nn.Conv2d(256, 512, kernel_size=3, padding=1),          # 64x64x512
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)                               # 32x32x512
        )
        # Decoder: reconstruct the image back to 1024x1024
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),          # 32x32x512
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),            # 64x64x512

            nn.Conv2d(512, 256, kernel_size=3, padding=1),          # 64x64x256
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),            # 128x128x256

            nn.Conv2d(256, 128, kernel_size=3, padding=1),          # 128x128x128
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),            # 256x256x128

            nn.Conv2d(128, 64, kernel_size=3, padding=1),           # 256x256x64
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),            # 512x512x64

            nn.Conv2d(64, 32, kernel_size=3, padding=1),            # 512x512x32
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),            # 1024x1024x32

            nn.Conv2d(32, IMG_CHANNELS, kernel_size=3, padding=1),   # 1024x1024x3
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def plot_and_save_reconstructions(original, reconstructed, n=10, save_path="reconstructed_vs_original.png"):
    """
    Plot original, reconstructed, and residual heatmap images side by side.
    'original' and 'reconstructed' are numpy arrays in shape (N, channels, H, W);
    they are converted to (H, W, channels) for plotting.
    """
    plt.figure(figsize=(15, 9))
    for i in range(n):
        # Convert from CHW to HWC for visualization
        orig_img = np.transpose(original[i], (1, 2, 0))
        recon_img = np.transpose(reconstructed[i], (1, 2, 0))
        
        # Original image (row 1)
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(orig_img)
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed image (row 2)
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(recon_img)
        plt.title("Reconstructed")
        plt.axis('off')
        
        # Residual heatmap (row 3)
        residual = np.abs(orig_img - recon_img)
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(residual, cmap="hot")
        plt.title("Residual")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_reconstruction_errors(model, images, device):
    """
    Compute the mean squared error (MSE) between the original images and their reconstructions.
    Returns a numpy array of errors (one per image) and the reconstructions.
    """
    model.eval()
    errors = []
    reconstructions = []
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.cpu().numpy()
        images_np = images.cpu().numpy()
        for orig, recon in zip(images_np, outputs):
            mse = np.mean((orig - recon) ** 2)
            errors.append(mse)
            reconstructions.append(recon)
    return np.array(errors), np.array(reconstructions)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Update these paths to point to your MVTec dataset directories.
    train_folder = "./carpet/train/good"  # e.g., "./mvtec/train/normal"
    test_folder = "./carpet/test/cut"      # e.g., "./mvtec/test"
    
    # Load training and test images
    print("Loading training images...")
    train_images_np = load_images_from_folder(train_folder)
    print("Training images shape:", train_images_np.shape)
    
    print("Loading test images...")
    test_images_np = load_images_from_folder(test_folder)
    print("Test images shape:", test_images_np.shape)
    
    # Convert numpy arrays to torch tensors
    x_train = torch.tensor(train_images_np, dtype=torch.float32)
    x_test = torch.tensor(test_images_np, dtype=torch.float32)
    
    # Create a TensorDataset and split training data into training and validation sets (90%/10%)
    dataset = TensorDataset(x_train, x_train)  # Using the same tensor for input and target
    total_samples = len(dataset)
    n_val = int(0.1 * total_samples)
    n_train = total_samples - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    batch_size = 16  # Adjust based on your hardware
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the autoencoder model
    model = Autoencoder().to(device)
    print(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 30
    patience = 5
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    # Training loop with early stopping based on validation loss
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
    
    # Compute reconstruction errors on the test set
    test_errors, reconstructions = compute_reconstruction_errors(model, x_test, device)
    print("Reconstruction errors on test set:", test_errors)
    
    # Determine anomaly threshold based on training reconstruction errors
    train_errors, _ = compute_reconstruction_errors(model, x_train, device)
    threshold = np.mean(train_errors) + 2 * np.std(train_errors)
    print("Anomaly detection threshold:", threshold)
    
    # Flag images with reconstruction error above the threshold as anomalies
    anomalies = test_errors > threshold
    print("Number of anomalies detected:", np.sum(anomalies))
    
    # Visualize original vs. reconstructed images along with residual heatmaps,
    # and save the figure to a file.
    plot_and_save_reconstructions(x_test.cpu().numpy(), reconstructions,
                                  n=10, save_path="pytorch_autoencoder_reconstructions.png")

if __name__ == "__main__":
    main()
