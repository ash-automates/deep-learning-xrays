"""
CNN model with transfer learning for chest X-ray classification.
Implements multiple architectures: ResNet, DenseNet, EfficientNet.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0
from tensorflow.keras.preprocessing import image
import numpy as np


class ChestXRayClassifier:
    """CNN classifier for chest X-ray disease detection using transfer learning."""
    
    def __init__(self, model_name='resnet50', num_classes=4, input_shape=(224, 224, 3)):
        """
        Initialize the classifier.
        
        Args:
            model_name (str): Base model name ('resnet50', 'densenet121', 'efficientnetb0')
            num_classes (int): Number of disease classes
            input_shape (tuple): Input image shape
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None
    
    def build_model(self, freeze_base=True):
        """
        Build the transfer learning model.
        
        Args:
            freeze_base (bool): Whether to freeze base model weights
            
        Returns:
            tf.keras.Model: Compiled model
        """
        # Load pre-trained base model
        if self.model_name == 'resnet50':
            base_model = ResNet50(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif self.model_name == 'densenet121':
            base_model = DenseNet121(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif self.model_name == 'efficientnetb0':
            base_model = EfficientNetB0(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Freeze base model layers
        if freeze_base:
            base_model.trainable = False
        
        # Build custom head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def get_model_summary(self):
        """
        Print model summary.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.summary()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
        """
        Train the model.
        
        Args:
            X_train (np.ndarray): Training images
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation images
            y_val (np.ndarray): Validation labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
            
        Returns:
            History: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Convert labels to one-hot encoding
        from tensorflow.keras.utils import to_categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.history
    
    def predict(self, image_array):
        """
        Make prediction on an image.
        
        Args:
            image_array (np.ndarray): Image array
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Add batch dimension if needed
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        return self.model.predict(image_array, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.ndarray): Test images
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        from tensorflow.keras.utils import to_categorical
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        
        results = self.model.evaluate(X_test, y_test_cat, verbose=0)
        
        return {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
    
    def save_model(self, path):
        """
        Save model to disk.
        
        Args:
            path (str): Path to save model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.save(path)
    
    def load_model(self, path):
        """
        Load model from disk.
        
        Args:
            path (str): Path to load model from
        """
        self.model = tf.keras.models.load_model(path)
