#!/usr/bin/env python3
"""
Experiment 3: Feature-Enriched Classification Based on Peak-Centered PPG Windows
==============================================================================

Objective:
To classify blood glucose levels based on structured feature vectors extracted from PPG 
windows centered around individual heartbeats. This experiment aims to simplify the input 
format using handcrafted features and test a classification-based framing of the BGL 
prediction task.

Key Differences from Experiment 4:
- Data balancing AFTER train/test split (vs before in Exp 4)
- Test set keeps real-world class imbalance
- 10 seconds before BGL timestamp to avoid data leakage
- Feature extraction: HR, AUC, FW, FW_25, FW_50, FW_75

Dataset & Preprocessing:
- Source: VitalDB
- Sampling Rate: 100 Hz 
- Window Size: 1 second (100 samples)
- Target: Extract 64 valid heartbeat-centered windows before each BGL timestamp
- Final format: 64 windows × 106 features (100 PPG + 6 extracted features)
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
    MAX_BGL = 200        # Filter out extreme values (200 for Exp 3 vs 250 for Exp 4)
    TIME_OFFSET = 10     # Use signal 10 seconds before BGL timestamp
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Paths
    BGL_DATA_PATH = "cleaned_bgl_data.parquet"
    PPG_DATA_DIR = "output/ppg_filtered_v3/"
    RESULTS_DIR = "experiments/experiment_3_results/"
    
    # Random seed for reproducibility
    RANDOM_SEED = 42

def setup_results_directory():
    """Create results directory structure"""
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.RESULTS_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(Config.RESULTS_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(Config.RESULTS_DIR, "data"), exist_ok=True)
    
def log_experiment_info(info_dict):
    """Log experiment information to JSON file"""
    log_file = os.path.join(Config.RESULTS_DIR, "experiment_3_log.json")
    with open(log_file, 'w') as f:
        json.dump(info_dict, f, indent=2)
    print(f"✅ Experiment log saved to: {log_file}")

class CNNBinaryClassification(nn.Module):
    """CNN model for binary classification with 64x106 input"""
    def __init__(self):
        super(CNNBinaryClassification, self).__init__()
        
        # CNN layers for processing 106-feature windows
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
        
        # Calculate output size: (106-2)/2 -> (52-2)/2 -> (25-2)/2 = 11.5 -> 11
        conv_output_size = 11
        
        self.fc = nn.Sequential(
            nn.Linear(128 * conv_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # Binary Classification
        )
        
    def forward(self, x):
        batch_size, sequence_length, features = x.shape
        
        # Process each window through CNN
        x = x.view(batch_size * sequence_length, 1, features)  # (B*64, 1, 106)
        x = self.cnn(x)  # → (B*64, 128, 11)
        
        # Flatten and average across windows
        x = x.view(x.size(0), -1)  # Flatten to (B*64, 128*11)
        x = x.view(batch_size, sequence_length, -1)  # → (B, 64, 128*11)
        
        # Average pooling across 64 windows
        x = torch.mean(x, dim=1)  # → (B, 128*11)
        
        # Classification
        x = self.fc(x)
        return x

def extract_ppg_features(window, fs=100):
    """Extract 6 features from PPG window: HR, AUC, FW, FW_25, FW_50, FW_75"""
    try:
        # Find peaks for HR calculation
        peak_dict = nk.ppg_findpeaks(window, sampling_rate=fs)
        peaks = peak_dict["PPG_Peaks"]
        
        # Calculate features
        HR = 60 / (np.mean(np.diff(peaks)) / fs) if len(peaks) > 1 else 0
        AUC = np.trapezoid(window)  # Area under curve
        FW = np.sum(window > 0)     # Number of positive values
        
        # Full width at different thresholds
        peak_amplitude = np.max(window)
        FW_25 = np.sum(window > 0.25 * peak_amplitude)
        FW_50 = np.sum(window > 0.50 * peak_amplitude)
        FW_75 = np.sum(window > 0.75 * peak_amplitude)
        
        return np.array([HR, AUC, FW, FW_25, FW_50, FW_75])
    except:
        return np.zeros(6)

def extract_peak_centered_windows(limit_cases=None):
    """
    Extract peak-centered PPG windows with feature enrichment
    
    Steps:
    1. Use signal 10 seconds before BGL timestamp (avoid data leakage)
    2. Find PPG peaks using neurokit2
    3. Extract 1-second windows centered on peaks
    4. Keep only windows with exactly one peak
    5. Extract 6 features per window
    6. Combine PPG signal + features (106 total features)
    """
    print(f"\n🔄 Processing Experiment 3 data (Feature-Enriched Classification)...")
    
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
    for case_id in tqdm(case_ids, desc="Processing Peak-Centered Windows"):
        try:
            case_id_int = int(case_id)
            ppg_file_path = os.path.join(Config.PPG_DATA_DIR, f"{case_id_int}.npy")
            
            if not os.path.exists(ppg_file_path):
                continue
                
            ppg_signal = np.load(ppg_file_path)
            case_bgl = bgl_data[bgl_data['caseid'] == case_id]
            
            for _, row in case_bgl.iterrows():
                timestamp, glucose = row['dt'], row['glucose']
                
                # Use signal 10 seconds before BGL timestamp to avoid data leakage
                signal_end_time = timestamp - Config.TIME_OFFSET
                if signal_end_time < 0:
                    continue
                
                end_idx = int(signal_end_time * Config.SAMPLING_RATE)
                
                # Ensure minimum 3 seconds of data
                if len(ppg_signal[:end_idx]) < 3 * Config.SAMPLING_RATE:
                    continue
                
                # Peak detection in available segment
                available_signal = ppg_signal[:end_idx]
                peak_dict = nk.ppg_findpeaks(available_signal, sampling_rate=Config.SAMPLING_RATE)
                peaks = peak_dict["PPG_Peaks"]
                
                # Extract windows centered around peaks
                valid_windows = []
                for peak_idx in reversed(peaks):  # Start from most recent peak
                    if len(valid_windows) >= Config.NUM_WINDOWS:
                        break
                        
                    # Center window around peak
                    start_w = peak_idx - (Config.WINDOW_SIZE // 2)
                    end_w = peak_idx + (Config.WINDOW_SIZE // 2)
                    
                    if start_w < 0 or end_w > len(available_signal):
                        continue
                        
                    window = available_signal[start_w:end_w]
                    
                    # Check for exactly one peak in this window
                    window_peak_dict = nk.ppg_findpeaks(window, sampling_rate=Config.SAMPLING_RATE)
                    window_peaks = window_peak_dict["PPG_Peaks"]
                    
                    if len(window_peaks) == 1:
                        # Extract 6 features from the window
                        features = extract_ppg_features(window, Config.SAMPLING_RATE)
                        
                        # Combine PPG signal (100) + features (6) = 106 total
                        window_with_features = np.concatenate([window, features])
                        valid_windows.append(window_with_features)
                
                # Only keep samples with exactly 64 valid windows
                if len(valid_windows) == Config.NUM_WINDOWS:
                    X_data.append(np.array(valid_windows))
                    Y_data.append(glucose)
                    
        except Exception as e:
            print(f"❌ Error processing case {case_id}: {e}")
    
    # Convert to arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    
    # Apply standardization across all feature dimensions
    if len(X_data) > 0:
        print(f"📊 Applying StandardScaler to {X_data.shape} data...")
        X_reshaped = X_data.reshape(-1, X_data.shape[-1])  # (N*64, 106)
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X_reshaped)
        X_data = X_standardized.reshape(X_data.shape)  # Back to (N, 64, 106)
    
    print(f"✅ Extracted data shape: X={X_data.shape}, Y={Y_data.shape}")
    return X_data, Y_data

def balance_after_split(X_train, Y_train, random_state=42):
    """
    Balance training data AFTER train/test split
    Strategy: Oversample Low (2x) + Undersample Not Low to match oversampled Low count
    """
    np.random.seed(random_state)
    
    # Find indices for each class
    low_indices = np.where(Y_train == 1)[0]   # Low glucose
    not_low_indices = np.where(Y_train == 0)[0]  # Not low glucose
    
    low_count = len(low_indices)
    not_low_count = len(not_low_indices)
    
    print(f"Original training distribution - Low: {low_count}, Not Low: {not_low_count}")
    
    # Oversample Low class to 2x its size
    low_oversampled = np.random.choice(low_indices, size=low_count * 2, replace=True)
    
    # Undersample Not Low class to match oversampled Low count
    target_not_low_count = len(low_oversampled)
    not_low_undersampled = np.random.choice(not_low_indices, size=target_not_low_count, replace=False)
    
    # Combine and shuffle
    balanced_indices = np.concatenate([low_oversampled, not_low_undersampled])
    np.random.shuffle(balanced_indices)
    
    X_balanced = X_train[balanced_indices]
    Y_balanced = Y_train[balanced_indices]
    
    print(f"Balanced training distribution - Low: {np.sum(Y_balanced == 1)}, Not Low: {np.sum(Y_balanced == 0)}")
    
    return X_balanced, Y_balanced

def prepare_data_experiment_3(X_data, Y_data):
    """
    Prepare data following Experiment 3 strategy:
    1. Filter extreme values (BGL > 200)
    2. Split into train/val/test (stratified)
    3. Balance ONLY training data (keep test unbalanced)
    4. Return balanced train, val, and unbalanced test
    """
    # Create binary labels
    Y_binary = np.where(Y_data < Config.BGL_THRESHOLD, 1, 0)  # 1 for Low, 0 for Not Low
    
    # Filter extreme values (BGL > 200)
    mask = Y_data <= Config.MAX_BGL
    X_filtered = X_data[mask]
    Y_filtered = Y_binary[mask]
    
    print(f"After filtering (BGL <= {Config.MAX_BGL}): {X_filtered.shape[0]} samples")
    print(f"Label distribution - Low: {np.sum(Y_filtered == 1)}, Not Low: {np.sum(Y_filtered == 0)}")
    
    # Split into train/val (90%) and test (10%) - stratified
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X_filtered, Y_filtered, 
        test_size=0.1, 
        random_state=Config.RANDOM_SEED, 
        stratify=Y_filtered
    )
    
    print(f"Test set (unbalanced) - Low: {np.sum(Y_test == 1)}, Not Low: {np.sum(Y_test == 0)}")
    
    # Balance ONLY the training/validation data
    X_balanced, Y_balanced = balance_after_split(X_trainval, Y_trainval, Config.RANDOM_SEED)
    
    # Split balanced data into train (80%) and val (20%)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_balanced, Y_balanced,
        test_size=0.2,
        random_state=Config.RANDOM_SEED,
        stratify=Y_balanced
    )
    
    print(f"Final data splits:")
    print(f"  Train: {X_train.shape[0]} samples (balanced)")
    print(f"  Val: {X_val.shape[0]} samples (balanced)")  
    print(f"  Test: {X_test.shape[0]} samples (unbalanced - real world distribution)")
    
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

def train_model(model, train_loader, val_loader):
    """Train the CNN model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    print(f"\n🚀 Training Experiment 3 model...")
    
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
            model_path = os.path.join(Config.RESULTS_DIR, "models", "best_cnn_model.pth")
            torch.save(model.state_dict(), model_path)
        
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss
    }

def evaluate_model(model, test_loader):
    """Evaluate model performance on unbalanced test set"""
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
    
    print(f"\n📊 Experiment 3 Test Results (Unbalanced Real-World Distribution):")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    return results

def plot_training_history(history):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Training Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Experiment 3 - Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['val_accuracies'], label='Validation Accuracy', color='green')
    ax2.set_title('Experiment 3 - Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(Config.RESULTS_DIR, "plots", "training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Training plot saved: {plot_path}")

def plot_confusion_matrix(conf_matrix):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title('Experiment 3 - Confusion Matrix (Unbalanced Test Set)')
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
    
    plot_path = os.path.join(Config.RESULTS_DIR, "plots", "confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎯 Confusion matrix plot saved: {plot_path}")

def main():
    """Main experiment runner"""
    print("🧪 EXPERIMENT 3: Feature-Enriched Classification Based on Peak-Centered PPG Windows")
    print("=" * 90)
    
    # Setup
    setup_results_directory()
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    start_time = datetime.now()
    
    try:
        # For development/testing, you can limit the number of cases
        limit_cases = None  # Set to e.g., 100 for testing
        
        # Extract peak-centered windows with features
        X_data, Y_data = extract_peak_centered_windows(limit_cases)
        
        if len(X_data) == 0:
            raise ValueError("No data extracted")
        
        # Save raw data
        data_path = os.path.join(Config.RESULTS_DIR, "data", "raw_data.npz")
        np.savez(data_path, X=X_data, Y=Y_data)
        print(f"💾 Raw data saved: {data_path}")
        
        # Prepare data (filter, split, balance)
        X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data_experiment_3(X_data, Y_data)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, X_val, X_test, Y_train, Y_val, Y_test
        )
        
        # Initialize model
        model = CNNBinaryClassification()
        print(f"🏗️  Model initialized for 64x106 input format")
        
        # Train model
        history = train_model(model, train_loader, val_loader)
        
        # Load best model for evaluation
        best_model_path = os.path.join(Config.RESULTS_DIR, "models", "best_cnn_model.pth")
        model.load_state_dict(torch.load(best_model_path))
        
        # Evaluate on unbalanced test set
        results = evaluate_model(model, test_loader)
        
        # Generate plots
        plot_training_history(history)
        plot_confusion_matrix(np.array(results['confusion_matrix']))
        
        # Save final trained model (matching expected filename)
        final_model_path = os.path.join(Config.RESULTS_DIR, "models", "cnn_binary_classification.pth")
        torch.save(model.state_dict(), final_model_path)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile experiment info
        experiment_info = {
            'experiment_name': 'Experiment 3: Feature-Enriched Classification',
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'config': {
                'sampling_rate': Config.SAMPLING_RATE,
                'window_size': Config.WINDOW_SIZE,
                'num_windows': Config.NUM_WINDOWS,
                'bgl_threshold': Config.BGL_THRESHOLD,
                'max_bgl': Config.MAX_BGL,
                'time_offset': Config.TIME_OFFSET,
                'batch_size': Config.BATCH_SIZE,
                'epochs': Config.EPOCHS,
                'learning_rate': Config.LEARNING_RATE,
                'random_seed': Config.RANDOM_SEED
            },
            'data_info': {
                'total_samples': int(len(X_data)),
                'input_shape': list(X_data.shape[1:]),  # [64, 106]
                'train_samples': int(len(X_train)),
                'val_samples': int(len(X_val)),
                'test_samples': int(len(X_test)),
                'balancing_strategy': 'after_split',
                'test_set': 'unbalanced_real_world_distribution'
            },
            'training_history': history,
            'test_results': results,
            'files': {
                'raw_data': data_path,
                'best_model': best_model_path,
                'final_model': final_model_path,
                'training_plot': os.path.join(Config.RESULTS_DIR, "plots", "training_history.png"),
                'confusion_matrix_plot': os.path.join(Config.RESULTS_DIR, "plots", "confusion_matrix.png")
            },
            'key_findings': {
                'feature_extraction': 'HR, AUC, FW, FW_25, FW_50, FW_75',
                'window_strategy': 'peak_centered_10s_before_bgl',
                'balancing_approach': 'balance_after_split_keep_test_unbalanced',
                'test_accuracy': results['accuracy'],
                'test_roc_auc': results['roc_auc']
            }
        }
        
        # Log experiment
        log_experiment_info(experiment_info)
        
        print(f"\n✅ Experiment 3 completed successfully!")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"📁 Results saved in: {Config.RESULTS_DIR}")
        
        return experiment_info
        
    except Exception as e:
        print(f"❌ Error in Experiment 3: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()