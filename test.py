import os

# Set image dimensions for 1024x1024 RGB images
IMG_HEIGHT, IMG_WIDTH = 1024, 1024
IMG_CHANNELS = 3

def load_images_from_folder(folder, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Load and preprocess images from a given folder.
    Images are resized to target_size and normalized to [0,1].
    """
    for filename in os.listdir(folder):
        print(filename)
        
def main():
    # Update these paths to point to your MVTec dataset directories.
    # The training folder should contain only normal images.
    train_folder = "./carpet/train/good"  # e.g., "./mvtec/train/normal"
    
    # Load training images
    print("Loading training images...")
    x_train = load_images_from_folder(train_folder)

main()