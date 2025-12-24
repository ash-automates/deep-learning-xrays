"""
Main training and evaluation script for chest X-ray disease classification.
Demonstrates the complete pipeline: data preparation, model training, and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, roc_auc_score
)
import seaborn as sns
from preprocessing import ChestXRayPreprocessor, create_synthetic_dataset
from model import ChestXRayClassifier


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history.
    
    Args:
        history: Training history object
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        classes (list): Class names
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curves(y_test, y_pred_proba, classes, save_path='roc_curves.png'):
    """
    Plot ROC curves for each class.
    
    Args:
        y_test (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        classes (list): Class names
        save_path (str): Path to save the plot
    """
    from sklearn.preprocessing import label_binarize
    
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        if len(classes) > 2:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        else:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, i])
        
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.close()


def main():
    """Main training and evaluation pipeline."""
    
    print("=" * 60)
    print("Chest X-Ray Disease Classification - Proof of Concept")
    print("=" * 60)
    
    # Configuration
    MODEL_NAME = 'resnet50'  # Options: 'resnet50', 'densenet121', 'efficientnetb0'
    NUM_CLASSES = 4
    EPOCHS = 3
    BATCH_SIZE = 32
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Step 1: Data Preparation
    print("\n" + "=" * 60)
    print("Step 1: Data Preparation")
    print("=" * 60)
    
    print("Creating synthetic dataset for demonstration...")
    X_data, y_data, classes = create_synthetic_dataset(num_samples=50)
    print(f"Dataset shape: {X_data.shape}")
    print(f"Classes: {classes}")
    
    # Split data into train, validation, and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Step 2: Build Model
    print("\n" + "=" * 60)
    print("Step 2: Building Model")
    print("=" * 60)
    
    classifier = ChestXRayClassifier(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        input_shape=(224, 224, 3)
    )
    
    print(f"Building {MODEL_NAME} model with transfer learning...")
    classifier.build_model(freeze_base=True)
    classifier.get_model_summary()
    
    # Step 3: Train Model
    print("\n" + "=" * 60)
    print("Step 3: Training Model")
    print("=" * 60)
    
    print("Training model...")
    history = classifier.train(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Plot training history
    plot_training_history(history)
    
    # Step 4: Evaluate Model
    print("\n" + "=" * 60)
    print("Step 4: Model Evaluation")
    print("=" * 60)
    
    print("Evaluating on test set...")
    metrics = classifier.evaluate(X_test, y_test)
    
    print("\nTest Set Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    # Make predictions
    y_pred_proba = classifier.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes)
    
    # Plot ROC curves
    plot_roc_curves(y_test, y_pred_proba, classes)
    
    # Step 5: Save Model
    print("\n" + "=" * 60)
    print("Step 5: Saving Model")
    print("=" * 60)
    
    model_path = f'{MODEL_NAME}_chest_xray_classifier.h5'
    classifier.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Step 6: Demonstration on New Data
    print("\n" + "=" * 60)
    print("Step 6: Prediction on New Data")
    print("=" * 60)
    
    # Predict on a random test sample
    sample_idx = 0
    sample_image = X_test[sample_idx:sample_idx+1]
    prediction = classifier.predict(sample_image)
    predicted_class = classes[np.argmax(prediction)]
    true_class = classes[y_test[sample_idx]]
    confidence = np.max(prediction) * 100
    
    print(f"\nSample Prediction:")
    print(f"  True class: {true_class}")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"  All probabilities: {[f'{classes[i]}: {prediction[0][i]*100:.2f}%' for i in range(NUM_CLASSES)]}")
    
    print("\n" + "=" * 60)
    print("Proof of Concept Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
