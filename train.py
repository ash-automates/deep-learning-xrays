"""
Main training and evaluation script for chest X-ray disease classification.
Demonstrates the complete pipeline: data preparation, model training, and evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import tensorflow as tf
from preprocessing import ChestXRayPreprocessor, create_synthetic_dataset
from model import ChestXRayClassifier


def load_dataset_from_directory(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    AUTOTUNE = tf.data.AUTOTUNE
    normalize = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)

    train_ds = train_ds.map(normalize).cache().shuffle(1024).prefetch(AUTOTUNE)
    val_ds = val_ds.map(normalize).cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, train_ds.class_names


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
    
    # Configuration (overridable via env)
    MODEL_NAME = os.getenv('MODEL_NAME', 'resnet50')  # Options: 'resnet50', 'densenet121', 'efficientnetb0'
    DATASET_DIR = os.getenv('DATASET_DIR')
    EPOCHS = int(os.getenv('EPOCHS', 3))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    if DATASET_DIR:
        print(f"  Dataset dir: {DATASET_DIR}")

    # Step 1: Data Preparation
    print("\n" + "=" * 60)
    print("Step 1: Data Preparation")
    print("=" * 60)

    using_directory = DATASET_DIR and os.path.isdir(DATASET_DIR)
    if using_directory:
        print("Loading dataset from directory (expects subfolders per class)...")
        train_ds, val_ds, classes = load_dataset_from_directory(DATASET_DIR, img_size=(224, 224), batch_size=BATCH_SIZE)
        NUM_CLASSES = len(classes)
        print(f"Classes: {classes}")
        print("Using validation set as test set for evaluation metrics.")
    else:
        print("DATASET_DIR not set or invalid; creating synthetic dataset for demonstration...")
        X_data, y_data, classes = create_synthetic_dataset(num_samples=50)
        NUM_CLASSES = len(classes)
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
    if using_directory:
        history = classifier.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            verbose=1,
        )
        test_source = val_ds
    else:
        # Compute class weights to handle imbalance in synthetic data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        print(f"Class weights: {class_weight_dict}")
        
        from tensorflow.keras.utils import to_categorical
        y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        history = classifier.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=[early_stopping],
            verbose=1
        )
        test_source = (X_test, y_test)

    # Plot training history
    plot_training_history(history)

    # Step 4: Evaluate Model
    print("\n" + "=" * 60)
    print("Step 4: Model Evaluation")
    print("=" * 60)

    print("Evaluating on test set...")
    if using_directory:
        loss, acc, prec, rec = classifier.model.evaluate(test_source, verbose=0)
        y_true = []
        y_pred_proba_batches = []
        for batch_x, batch_y in test_source:
            preds = classifier.model.predict(batch_x, verbose=0)
            y_pred_proba_batches.append(preds)
            y_true.append(batch_y)
        y_pred_proba = np.vstack(y_pred_proba_batches)
        y_true = np.vstack(y_true)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        metrics = {
            'loss': loss,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
        }
    else:
        metrics = classifier.evaluate(*test_source)
        y_pred_proba = classifier.predict(test_source[0])
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true_labels = test_source[1]

    print("\nTest Set Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred, target_names=classes))

    # Plot confusion matrix
    plot_confusion_matrix(y_true_labels, y_pred, classes)

    # Plot ROC curves
    plot_roc_curves(y_true_labels, y_pred_proba, classes)

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

    if using_directory:
        sample_batch = next(iter(test_source.take(1)))
        sample_image, sample_label = sample_batch
        sample_pred = classifier.model.predict(sample_image[:1], verbose=0)
        prediction = sample_pred
        predicted_class = classes[np.argmax(prediction)]
        true_class = classes[np.argmax(sample_label[0].numpy())]
    else:
        sample_idx = 0
        sample_image = test_source[0][sample_idx:sample_idx+1]
        prediction = classifier.predict(sample_image)
        predicted_class = classes[np.argmax(prediction)]
        true_class = classes[test_source[1][sample_idx]]

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
