import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set image dimensions for 1024x1024 RGB images
IMG_HEIGHT, IMG_WIDTH = 1024, 1024
IMG_CHANNELS = 3

def load_images_from_folder(folder, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Load and preprocess images from a given folder.
    Images are resized to target_size and normalized to [0,1].
    """
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = load_img(img_path, target_size=target_size)
            img = img_to_array(img) / 255.0
            images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(images)

def build_autoencoder(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    """
    Build a deeper convolutional autoencoder that compresses the image down further.
    """
    input_img = Input(shape=input_shape)
    
    # Encoder: progressively reduce spatial dimensions
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)       # 1024x1024x32
    x = MaxPooling2D((2, 2), padding='same')(x)                                  # 512x512x32
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)                # 512x512x64
    x = MaxPooling2D((2, 2), padding='same')(x)                                  # 256x256x64
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)               # 256x256x128
    x = MaxPooling2D((2, 2), padding='same')(x)                                  # 128x128x128
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)               # 128x128x256
    x = MaxPooling2D((2, 2), padding='same')(x)                                  # 64x64x256
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)               # 64x64x512
    encoded = MaxPooling2D((2, 2), padding='same')(x)                           # 32x32x512
    
    # Decoder: reconstruct the image back to 1024x1024
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(encoded)         # 32x32x512
    x = UpSampling2D((2, 2))(x)                                                  # 64x64x512
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)               # 64x64x256
    x = UpSampling2D((2, 2))(x)                                                  # 128x128x256
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)               # 128x128x128
    x = UpSampling2D((2, 2))(x)                                                  # 256x256x128
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)                # 256x256x64
    x = UpSampling2D((2, 2))(x)                                                  # 512x512x64
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)                # 512x512x32
    x = UpSampling2D((2, 2))(x)                                                  # 1024x1024x32
    
    decoded = Conv2D(IMG_CHANNELS, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    autoencoder.summary()
    return autoencoder

def compute_reconstruction_errors(model, images):
    """
    Compute mean squared error (MSE) between the original images and their reconstructions.
    """
    reconstructions = model.predict(images)
    errors = np.mean(np.power(images - reconstructions, 2), axis=(1, 2, 3))
    return errors, reconstructions

def main():
    # Update these paths to point to your MVTec dataset directories.
    # The training folder should contain only normal images.
    train_folder = "./carpet/train/good"  # e.g., "./mvtec/train/normal"
    test_folder = "./carpet/test/cut"      # e.g., "./mvtec/test"
    
    # Load training images
    print("Loading training images...")
    x_train = load_images_from_folder(train_folder)
    print("Training images shape:", x_train.shape)
    
    # Load test images
    print("Loading test images...")
    x_test = load_images_from_folder(test_folder)
    print("Test images shape:", x_test.shape)
    
    # Build the deep convolutional autoencoder
    autoencoder = build_autoencoder()
    
    # Train the autoencoder on normal images
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    history = autoencoder.fit(
        x_train, x_train,
        epochs=30,
        batch_size=16,  # Adjust batch size if needed given the large image size
        shuffle=True,
        validation_split=0.1,
        callbacks=[early_stop]
    )
    
    # Evaluate reconstruction error on the test set
    test_errors, reconstructions = compute_reconstruction_errors(autoencoder, x_test)
    print("Reconstruction errors on test set:", test_errors)
    
    # Determine an anomaly threshold based on training reconstruction errors.
    train_errors, _ = compute_reconstruction_errors(autoencoder, x_train)
    threshold = np.mean(train_errors) + 2 * np.std(train_errors)
    print("Anomaly detection threshold:", threshold)
    
    # Flag images with reconstruction error above the threshold as anomalies
    anomalies = test_errors > threshold
    print("Number of anomalies detected:", np.sum(anomalies))
    
if __name__ == "__main__":
    main()
