"""
Image preprocessing module for chest X-ray classification.
Implements normalization, resizing, and data augmentation techniques.
"""

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ChestXRayPreprocessor:
    """Handles preprocessing of chest X-ray images."""
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size (tuple): Target size for images (height, width)
        """
        self.target_size = target_size
    
    def enhance_contrast(self, gray_image):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for X-ray enhancement.
        
        Args:
            gray_image (np.ndarray): Grayscale image
            
        Returns:
            np.ndarray: Enhanced grayscale image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_image)
    
    def normalize_image(self, image):
        """
        Normalize image to [0, 1] range.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def resize_image(self, image, target_size=None):
        """
        Resize image to target size.
        
        Args:
            image (np.ndarray): Input image
            target_size (tuple): Target size (height, width)
            
        Returns:
            np.ndarray: Resized image
        """
        if target_size is None:
            target_size = self.target_size
        return cv2.resize(image, (target_size[1], target_size[0]))
    
    def preprocess_image(self, image_path):
        """
        Complete preprocessing pipeline for a single image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Enhance contrast with CLAHE
        image = self.enhance_contrast(image)
        
        # Resize
        image = self.resize_image(image)
        
        # Convert to 3 channels (for ResNet compatibility)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Normalize
        image = self.normalize_image(image)
        
        return image
    
    def get_train_augmentation(self):
        """
        Get data augmentation generator for training with stronger augmentation.
        
        Returns:
            ImageDataGenerator: Configured augmentation generator
        """
        return ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.3,
            shear_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            rescale=1./255
        )
    
    def get_validation_augmentation(self):
        """
        Get data augmentation generator for validation (no augmentation, only rescaling).
        
        Returns:
            ImageDataGenerator: Configured generator for validation
        """
        return ImageDataGenerator(rescale=1./255)


def create_synthetic_dataset(num_samples=100):
    """
    Create a synthetic dataset for demonstration purposes.
    
    Args:
        num_samples (int): Number of samples to create per class
        
    Returns:
        tuple: (X_data, y_data) where X_data is images and y_data is labels
    """
    classes = ['healthy', 'viral_pneumonia', 'bacterial_pneumonia', 'covid19']
    X_data = []
    y_data = []
    
    for label_idx, class_name in enumerate(classes):
        for _ in range(num_samples):
            # Create synthetic 224x224 grayscale image
            synthetic_img = np.random.randint(50, 200, (224, 224, 1), dtype=np.uint8)
            
            # Add some patterns to differentiate classes
            if label_idx > 0:
                # Add some noise/patterns for non-healthy classes
                noise = np.random.normal(0, 20, (224, 224, 1))
                synthetic_img = np.clip(synthetic_img + noise, 0, 255).astype(np.uint8)
            
            # Convert to 3 channels
            synthetic_img = np.repeat(synthetic_img, 3, axis=2)
            
            X_data.append(synthetic_img.astype(np.float32) / 255.0)
            y_data.append(label_idx)
    
    return np.array(X_data), np.array(y_data), classes
