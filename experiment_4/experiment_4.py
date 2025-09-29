#!/usr/bin/env python3
"""
Experiment 4: Classification with Augmentation Before Splitting
==============================================================

Objective:
- Investigate how the timing of window extraction and the order of data balancing affect model performance
- Perform oversampling/undersampling before data splitting, in contrast to Experiment 3
- Compare two strategies for extracting 64 PPG windows:
  (A) Immediately before the BGL measurement
  (B) 10 seconds before the BGL measurement

Input Format:
- Variant A: 64 windows × 100 features per sample
- Variant B: 64 windows × 106 features per sample (100 PPG + 6 extracted features)
- Data standardized using StandardScaler

Configuration:
- Labeling threshold: BGL < 70 → Low
- Removal of extreme outliers: BGL > 250
- Oversampling Low and undersampling Not Low (2:1 ratio)
- CNN model architecture with binary classification
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configuration
class Config:
    # Data parameters
    SAMPLING_RATE = 100  # Hz
    WINDOW_SIZE = 100    # 1-second window
    NUM_WINDOWS = 64     # Number of windows to extract
    BGL_THRESHOLD = 70   # Low glucose threshold
    MAX_BGL = 250        # Filter out extreme values
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Paths
    BGL_DATA_PATH = "cleaned_bgl_data.parquet"
    PPG_DATA_DIR = "output/ppg_filtered_v3/"
    RESULTS_DIR = "experiments/experiment_4_results/"
    
    # Random seed for reproducibility
    RANDOM_SEED = 42

def setup_results_directory():
    """Create results directory structure"""
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.RESULTS_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(Config.RESULTS_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(Config.RESULTS_DIR, "data"), exist_ok=True)
    
def log_experiment_info(variant, info_dict):
    """Log experiment information to JSON file"""
    log_file = os.path.join(Config.RESULTS_DIR, f"experiment_log_variant_{variant}.json")
    with open(log_file, 'w') as f:
        json.dump(info_dict, f, indent=2)
    print(f"✅ Experiment log saved to: {log_file}")

class CNNBinaryClassification(nn.Module):
    """CNN model for binary classification"""
    def __init__(self, input_features=100):
        super(CNNBinaryClassification, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AvgPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AvgPool1d(kernel_size=2)
        )
        
        # Calculate the output size after convolutions
        # For input_features=100: (100-2)/2 -> (49-2)/2 -> (23-2)/2 = 10.5 -> 10
        # For input_features=106: (106-2)/2 -> (52-2)/2 -> (25-2)/2 = 11.5 -> 11
        conv_output_size = ((((input_features - 2) // 2) - 2) // 2 - 2) // 2
        
        self.fc = nn.Sequential(
            nn.Linear(128 * conv_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # Binary Classification
        )
        
    def forward(self, x):
        batch_size, sequence_length, features = x.shape
        
        # Reshape for CNN processing
        x = x.view(batch_size * sequence_length, 1, features)  # (B*64, 1, features)
        x = self.cnn(x)  # → (B*64, 128, conv_output_size)
        
        # Flatten and reshape back
        x = x.view(x.size(0), -1)  # Flatten to (B*64, 128*conv_output_size)
        x = x.view(batch_size, sequence_length, -1)  # → (B, 64, 128*conv_output_size)
        
        # Average pooling across windows
        x = torch.mean(x, dim=1)  # → (B, 128*conv_output_size)
        
        # Classification
        x = self.fc(x)
        return x

def extract_ppg_features(window, fs=100):
    """Extract additional features from PPG window"""
    try:
        # Find peaks
        peak_dict = nk.ppg_findpeaks(window, sampling_rate=fs)
        small_peaks = peak_dict["PPG_Peaks"]
        
        # Calculate features
        HR = 60 / (np.mean(np.diff(small_peaks)) / fs) if len(small_peaks) > 1 else 0
        AUC = np.trapezoid(window)
        FW = np.sum(window > 0)
        FW_25 = np.sum(window > 0.25 * np.max(window))
        FW_50 = np.sum(window > 0.50 * np.max(window))
        FW_75 = np.sum(window > 0.75 * np.max(window))
        
        return np.array([HR, AUC, FW, FW_25, FW_50, FW_75])
    except:
        return np.zeros(6)

def extract_windows_data(variant="A", limit_cases=None):
    """
    Extract windowed data for specified variant
    
    Args:
        variant: "A" (immediate) or "B" (10s before)
        limit_cases: Optional limit on number of cases to process
    """
    print(f"\n🔄 Processing Variant {variant} data...")
    
    # Load BGL data
    if not os.path.exists(Config.BGL_DATA_PATH):
        raise FileNotFoundError(f"BGL data not found at {Config.BGL_DATA_PATH}")
    
    bgl_data = pd.read_parquet(Config.BGL_DATA_PATH)
    case_ids = bgl_data['caseid'].unique()
    
    if limit_cases:
        case_ids = case_ids[:limit_cases]
        print(f"Processing limited to first {limit_cases} cases")
    
    X_data, Y_data = [], []
    
    # Process each case
    for case_id in tqdm(case_ids, desc=f"Processing Variant {variant}"):
        try:
            case_id_int = int(case_id)
            ppg_file_path = os.path.join(Config.PPG_DATA_DIR, f"{case_id_int}.npy")
            
            if not os.path.exists(ppg_file_path):
                continue
                
            ppg_signal = np.load(ppg_file_path)
            case_bgl = bgl_data[bgl_data['caseid'] == case_id]
            
            for _, row in case_bgl.iterrows():
                timestamp, glucose = row['dt'], row['glucose']
                
                # Apply timing variant
                if variant == "B":
                    timestamp = timestamp - 10  # 10 seconds before
                    if timestamp < 0:
                        continue
                
                end_idx = int(timestamp * Config.SAMPLING_RATE)
                
                if len(ppg_signal[:end_idx]) < 3 * Config.SAMPLING_RATE:
                    continue
                
                # Extract windows
                valid_windows = []
                peak_dict = nk.ppg_findpeaks(ppg_signal[:end_idx], sampling_rate=Config.SAMPLING_RATE)
                peaks = peak_dict["PPG_Peaks"]
                
                for peak_idx in reversed(peaks):
                    if len(valid_windows) >= Config.NUM_WINDOWS:
                        break
                        
                    start_w = peak_idx - (Config.WINDOW_SIZE // 2)
                    end_w = peak_idx + (Config.WINDOW_SIZE // 2)
                    
                    if start_w < 0 or end_w > len(ppg_signal):
                        continue
                        
                    window = ppg_signal[start_w:end_w]
                    
                    # Check for single peak
                    small_peak_dict = nk.ppg_findpeaks(window, sampling_rate=Config.SAMPLING_RATE)
                    small_peaks = small_peak_dict["PPG_Peaks"]
                    
                    if len(small_peaks) == 1:
                        if variant == "A":
                            # Variant A: Only standardized PPG signal
                            scaler = StandardScaler()
                            w_standardized = scaler.fit_transform(window.reshape(-1, 1)).flatten()
                            valid_windows.append(w_standardized)
                        else:
                            # Variant B: PPG signal + extracted features
                            features = extract_ppg_features(window, Config.SAMPLING_RATE)
                            w_with_features = np.concatenate([window, features])
                            valid_windows.append(w_with_features)
                
                if len(valid_windows) == Config.NUM_WINDOWS:
                    X_data.append(np.array(valid_windows))
                    Y_data.append(glucose)
                    
        except Exception as e:
            print(f"❌ Error processing case {case_id}: {e}")
    
    # Convert to arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    
    # Apply global standardization for Variant B
    if variant == "B" and len(X_data) > 0:
        X_reshaped = X_data.reshape(-1, X_data.shape[-1])
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X_reshaped).reshape(X_data.shape)
        X_data = X_standardized
    
    print(f"✅ Variant {variant} - Extracted data shape: X={X_data.shape}, Y={Y_data.shape}")
    return X_data, Y_data

def balance_binary_data(X, Y, random_state=42):
    """Balance binary classification data with oversampling and undersampling"""
    np.random.seed(random_state)
    
    # Find indices for each class
    pos_indices = np.where(Y == 1)[0]  # Low glucose
    neg_indices = np.where(Y == 0)[0]  # Not low glucose
    
    pos_count = len(pos_indices)
    neg_count = len(neg_indices)
    
    print(f"Original distribution - Low: {pos_count}, Not Low: {neg_count}")
    
    # Target ratio: 1:2 (Low:Not Low)
    desired_neg_count = pos_count * 2
    
    # Oversample positive class (Low)
    pos_oversampled = np.random.choice(pos_indices, size=pos_count * 2, replace=True)
    
    # Undersample negative class (Not Low)
    neg_undersampled = np.random.choice(neg_indices, size=desired_neg_count, replace=False)
    
    # Combine and shuffle
    combined_indices = np.concatenate([pos_oversampled, neg_undersampled])
    np.random.shuffle(combined_indices)
    
    X_balanced = X[combined_indices]
    Y_balanced = Y[combined_indices]
    
    print(f"Balanced distribution - Low: {np.sum(Y_balanced == 1)}, Not Low: {np.sum(Y_balanced == 0)}")
    
    return X_balanced, Y_balanced

def prepare_data(X_data, Y_data):
    """Prepare data for training with filtering and balancing"""
    # Define binary labels
    threshold = Config.BGL_THRESHOLD
    Y_binary = np.where(Y_data < threshold, 1, 0)  # 1 for Low, 0 for Not Low
    
    # Filter extreme values
    mask = (Y_data <= Config.MAX_BGL) | (Y_binary == 1)
    X_filtered = X_data[mask]
    Y_filtered = Y_binary[mask]
    
    print(f"After filtering: X={X_filtered.shape}, Y={Y_filtered.shape}")
    
    # Split into train/val and test first
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X_filtered, Y_filtered, test_size=0.1, random_state=Config.RANDOM_SEED, stratify=Y_filtered
    )
    
    # Balance the training/validation data
    X_balanced, Y_balanced = balance_binary_data(X_trainval, Y_trainval, Config.RANDOM_SEED)
    
    # Split balanced data into train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_balanced, Y_balanced, test_size=0.2, random_state=Config.RANDOM_SEED, stratify=Y_balanced
    )
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def create_data_loaders(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    """Create PyTorch data loaders"""
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, Y_train_tensor), 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, Y_val_tensor), 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, Y_test_tensor), 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, variant):
    """Train the model and return training history"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    print(f"\n🚀 Training Variant {variant} model...")
    
    for epoch in range(Config.EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for X_batch, Y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}"):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_val_batch, Y_val_batch in val_loader:
                X_val_batch, Y_val_batch = X_val_batch.to(device), Y_val_batch.to(device)
                val_outputs = model(X_val_batch)
                val_loss += criterion(val_outputs, Y_val_batch).item()
                
                _, predicted = torch.max(val_outputs, 1)
                total += Y_val_batch.size(0)
                correct += (predicted == Y_val_batch).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(Config.RESULTS_DIR, "models", f"best_model_variant_{variant}.pth")
            torch.save(model.state_dict(), model_path)
        
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss
    }

def evaluate_model(model, test_loader, variant):
    """Evaluate model performance"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            outputs = model(X_batch)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            
            predictions.extend(predicted)
            true_values.extend(Y_batch.cpu().numpy())
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    # Calculate metrics
    accuracy = accuracy_score(true_values, predictions)
    conf_matrix = confusion_matrix(true_values, predictions)
    report = classification_report(true_values, predictions, target_names=['Not Low', 'Low'], output_dict=True)
    roc_auc = roc_auc_score(true_values, predictions)
    
    results = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report,
        'predictions': predictions.tolist(),
        'true_values': true_values.tolist()
    }
    
    print(f"\n📊 Variant {variant} Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    return results

def plot_training_history(history, variant):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Training Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title(f'Variant {variant} - Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['val_accuracies'], label='Validation Accuracy', color='green')
    ax2.set_title(f'Variant {variant} - Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(Config.RESULTS_DIR, "plots", f"training_history_variant_{variant}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Training plot saved: {plot_path}")

def plot_confusion_matrix(conf_matrix, variant):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title(f'Variant {variant} - Confusion Matrix')
    plt.colorbar()
    
    classes = ['Not Low', 'Low']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = os.path.join(Config.RESULTS_DIR, "plots", f"confusion_matrix_variant_{variant}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎯 Confusion matrix plot saved: {plot_path}")

def run_variant_experiment(variant, limit_cases=None):
    """Run complete experiment for a variant"""
    print(f"\n{'='*60}")
    print(f"🧪 EXPERIMENT 4 - VARIANT {variant}")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    try:
        # Extract data
        X_data, Y_data = extract_windows_data(variant, limit_cases)
        
        if len(X_data) == 0:
            raise ValueError(f"No data extracted for Variant {variant}")
        
        # Save raw data
        data_path = os.path.join(Config.RESULTS_DIR, "data", f"raw_data_variant_{variant}.npz")
        np.savez(data_path, X=X_data, Y=Y_data)
        print(f"💾 Raw data saved: {data_path}")
        
        # Prepare data
        X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data(X_data, Y_data)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, X_val, X_test, Y_train, Y_val, Y_test
        )
        
        # Initialize model
        input_features = X_data.shape[-1]  # 100 for Variant A, 106 for Variant B
        model = CNNBinaryClassification(input_features)
        
        print(f"🏗️  Model initialized with {input_features} input features")
        
        # Train model
        history = train_model(model, train_loader, val_loader, variant)
        
        # Load best model for evaluation
        best_model_path = os.path.join(Config.RESULTS_DIR, "models", f"best_model_variant_{variant}.pth")
        model.load_state_dict(torch.load(best_model_path))
        
        # Evaluate model
        results = evaluate_model(model, test_loader, variant)
        
        # Generate plots
        plot_training_history(history, variant)
        plot_confusion_matrix(np.array(results['confusion_matrix']), variant)
        
        # Save final trained model
        final_model_path = os.path.join(Config.RESULTS_DIR, "models", f"final_model_variant_{variant}.pth")
        torch.save(model.state_dict(), final_model_path)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile experiment info
        experiment_info = {
            'variant': variant,
            'experiment_name': f'Experiment 4 - Variant {variant}',
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'config': {
                'sampling_rate': Config.SAMPLING_RATE,
                'window_size': Config.WINDOW_SIZE,
                'num_windows': Config.NUM_WINDOWS,
                'bgl_threshold': Config.BGL_THRESHOLD,
                'max_bgl': Config.MAX_BGL,
                'batch_size': Config.BATCH_SIZE,
                'epochs': Config.EPOCHS,
                'learning_rate': Config.LEARNING_RATE,
                'random_seed': Config.RANDOM_SEED
            },
            'data_info': {
                'total_samples': int(len(X_data)),
                'input_features': int(input_features),
                'train_samples': int(len(X_train)),
                'val_samples': int(len(X_val)),
                'test_samples': int(len(X_test))
            },
            'training_history': history,
            'test_results': results,
            'files': {
                'raw_data': data_path,
                'best_model': best_model_path,
                'final_model': final_model_path,
                'training_plot': os.path.join(Config.RESULTS_DIR, "plots", f"training_history_variant_{variant}.png"),
                'confusion_matrix_plot': os.path.join(Config.RESULTS_DIR, "plots", f"confusion_matrix_variant_{variant}.png")
            }
        }
        
        # Log experiment
        log_experiment_info(variant, experiment_info)
        
        print(f"\n✅ Variant {variant} experiment completed successfully!")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        
        return experiment_info
        
    except Exception as e:
        print(f"❌ Error in Variant {variant} experiment: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_variants(variant_a_info, variant_b_info):
    """Compare results between variants"""
    if not variant_a_info or not variant_b_info:
        print("❌ Cannot compare variants - missing experiment results")
        return
    
    print(f"\n{'='*60}")
    print("📊 VARIANT COMPARISON")
    print(f"{'='*60}")
    
    # Extract key metrics
    metrics_a = variant_a_info['test_results']
    metrics_b = variant_b_info['test_results']
    
    comparison = {
        'variant_a': {
            'accuracy': metrics_a['accuracy'],
            'roc_auc': metrics_a['roc_auc'],
            'precision_low': metrics_a['classification_report']['Low']['precision'],
            'recall_low': metrics_a['classification_report']['Low']['recall'],
            'f1_low': metrics_a['classification_report']['Low']['f1-score']
        },
        'variant_b': {
            'accuracy': metrics_b['accuracy'],
            'roc_auc': metrics_b['roc_auc'],
            'precision_low': metrics_b['classification_report']['Low']['precision'],
            'recall_low': metrics_b['classification_report']['Low']['recall'],
            'f1_low': metrics_b['classification_report']['Low']['f1-score']
        }
    }
    
    print(f"Variant A (Immediate):")
    print(f"  Accuracy: {comparison['variant_a']['accuracy']:.4f}")
    print(f"  ROC AUC:  {comparison['variant_a']['roc_auc']:.4f}")
    print(f"  Low Glucose - Precision: {comparison['variant_a']['precision_low']:.4f}, Recall: {comparison['variant_a']['recall_low']:.4f}, F1: {comparison['variant_a']['f1_low']:.4f}")
    
    print(f"\nVariant B (10s Before):")
    print(f"  Accuracy: {comparison['variant_b']['accuracy']:.4f}")
    print(f"  ROC AUC:  {comparison['variant_b']['roc_auc']:.4f}")
    print(f"  Low Glucose - Precision: {comparison['variant_b']['precision_low']:.4f}, Recall: {comparison['variant_b']['recall_low']:.4f}, F1: {comparison['variant_b']['f1_low']:.4f}")
    
    # Determine winner
    print(f"\n🏆 PERFORMANCE COMPARISON:")
    if comparison['variant_a']['accuracy'] > comparison['variant_b']['accuracy']:
        print(f"  Accuracy: Variant A wins ({comparison['variant_a']['accuracy']:.4f} vs {comparison['variant_b']['accuracy']:.4f})")
    else:
        print(f"  Accuracy: Variant B wins ({comparison['variant_b']['accuracy']:.4f} vs {comparison['variant_a']['accuracy']:.4f})")
        
    if comparison['variant_a']['roc_auc'] > comparison['variant_b']['roc_auc']:
        print(f"  ROC AUC: Variant A wins ({comparison['variant_a']['roc_auc']:.4f} vs {comparison['variant_b']['roc_auc']:.4f})")
    else:
        print(f"  ROC AUC: Variant B wins ({comparison['variant_b']['roc_auc']:.4f} vs {comparison['variant_a']['roc_auc']:.4f})")
    
    # Save comparison
    comparison_path = os.path.join(Config.RESULTS_DIR, "variant_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Comparison saved: {comparison_path}")

def main():
    """Main experiment runner"""
    print("🧪 EXPERIMENT 4: Classification with Augmentation Before Splitting")
    print("=" * 80)
    
    # Setup
    setup_results_directory()
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # For development/testing, you can limit the number of cases
    limit_cases = None  # Set to e.g., 100 for testing
    
    # Run experiments
    print("\n🚀 Starting experiments...")
    
    variant_a_info = run_variant_experiment("A", limit_cases)
    variant_b_info = run_variant_experiment("B", limit_cases)
    
    # Compare results
    if variant_a_info and variant_b_info:
        compare_variants(variant_a_info, variant_b_info)
    
    print(f"\n🎉 All experiments completed!")
    print(f"📁 Results saved in: {Config.RESULTS_DIR}")

if __name__ == "__main__":
    main()