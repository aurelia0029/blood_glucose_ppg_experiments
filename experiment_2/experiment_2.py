#!/usr/bin/env python3
"""
Experiment 2: Augmented MUST Dataset with CNN-GRU Model
======================================================

Objective:
To predict blood glucose levels (BGL) using the MUST dataset through a robust preprocessing 
pipeline, followed by targeted Gaussian noise augmentation and a CNN-GRU hybrid model. 
Unlike the flawed approach in the reference paper, this experiment ensures data integrity 
by performing data splitting before augmentation.

Key Features:
- MUST dataset (VitalDB) with 10-second PPG segments
- Data splitting BEFORE augmentation (prevents data leakage)
- Targeted Gaussian noise augmentation for underrepresented BGL ranges
- CNN-GRU hybrid architecture for regression
- MAE loss function for continuous BGL prediction

Dataset & Preprocessing:
- Source: VitalDB PPG signals (resampled to 100 Hz)
- Filtering: 4th-order Butterworth filter (0.5-4 Hz passband)
- Normalization: Min-Max scaling per signal (0-1 range)
- Segment Length: 10 seconds (1000 samples at 100 Hz)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.signal import butter, filtfilt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configuration
class Config:
    # Data parameters
    SAMPLING_RATE = 100    # Hz (downsampled from 2175 Hz)
    SEGMENT_LENGTH = 1000  # 10 seconds * 100 Hz
    BGL_MIN = 50           # Minimum BGL to include
    BGL_MAX = 150          # Maximum BGL to include
    
    # Augmentation parameters
    NOISE_LEVELS_GENERAL = [0.01, 0.05, 0.1]        # For BGL < 80
    NOISE_LEVELS_TARGETED = [0.05, 0.1, 0.15, 0.2]  # For BGL 40-60
    EXTRA_AUGMENTATIONS = 3                          # Extra copies for BGL 40-60
    MAX_SAMPLES_PER_RANGE = 100                      # Downsampling limit
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    
    # Model parameters
    GRU_HIDDEN_SIZE = 128
    GRU_LAYERS = 2
    DROPOUT = 0.5
    
    # Filter parameters (Butterworth)
    FILTER_ORDER = 4
    LOWCUT = 0.5   # Hz
    HIGHCUT = 4.0  # Hz
    
    # Paths
    BGL_DATA_PATH = "cleaned_bgl_data.parquet"
    PPG_DATA_DIR = "output/ppg_filtered_v3/"
    RESULTS_DIR = "experiments/experiment_2_results/"
    
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
    log_file = os.path.join(Config.RESULTS_DIR, "experiment_2_log.json")
    with open(log_file, 'w') as f:
        json.dump(info_dict, f, indent=2)
    print(f"✅ Experiment log saved to: {log_file}")

def apply_butterworth_filter(signal, lowcut, highcut, fs, order=4):
    """Apply Butterworth bandpass filter to remove noise and artifacts"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

class CNNGRUHybrid(nn.Module):
    """CNN-GRU Hybrid Model for BGL Regression"""
    def __init__(self, gru_hidden_size=128, gru_layers=2, dropout=0.5):
        super(CNNGRUHybrid, self).__init__()

        # CNN Block 1 (kernel size 3)
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # CNN Block 2 (kernel size 5)
        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # GRU Block (bidirectional)
        self.gru_block = nn.GRU(
            input_size=1, 
            hidden_size=gru_hidden_size, 
            num_layers=gru_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0
        )

        # Calculate CNN output sizes (for 1000 input length)
        # After conv+pool: 1000 -> 500 (each block)
        # CNN outputs: 64 channels * 500 length = 32000 features each
        cnn_output_size = 32000

        # Fully Connected Layers for each pathway
        self.fc1 = nn.Sequential(
            nn.Linear(cnn_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(cnn_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 128),  # Bidirectional GRU
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final regression layer (3 pathways combined)
        self.final_fc = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Single regression output
        )
        
    def forward(self, x):
        # Input Shape: (batch_size, 1000)
        batch_size = x.size(0)
        
        # Reshape for CNN input (batch_size, 1, 1000)
        x_cnn = x.unsqueeze(1)

        # CNN Pathway 1 (kernel size 3)
        cnn1_out = self.cnn_block1(x_cnn)
        cnn1_out = cnn1_out.view(batch_size, -1)  # Flatten

        # CNN Pathway 2 (kernel size 5)
        cnn2_out = self.cnn_block2(x_cnn)
        cnn2_out = cnn2_out.view(batch_size, -1)  # Flatten

        # GRU Pathway
        x_gru = x.unsqueeze(-1)  # (batch_size, 1000, 1)
        gru_out, _ = self.gru_block(x_gru)
        gru_out = gru_out[:, -1, :]  # Take last time step

        # Apply FC layers to each pathway
        cnn1_features = self.fc1(cnn1_out)
        cnn2_features = self.fc2(cnn2_out)
        gru_features = self.fc3(gru_out)

        # Combine all pathways
        combined = torch.cat((cnn1_features, cnn2_features, gru_features), dim=1)

        # Final regression output
        output = self.final_fc(combined)
        return output

def load_and_preprocess_data(limit_cases=None):
    """
    Load and preprocess PPG-BGL data following MUST dataset approach
    
    Steps:
    1. Load PPG signals and BGL data
    2. Extract 10-second segments before each BGL measurement  
    3. Apply Butterworth filtering
    4. Apply Min-Max normalization per signal
    5. Filter by BGL range
    """
    print(f"\n🔄 Loading and preprocessing MUST dataset...")
    
    # Load BGL data
    if not os.path.exists(Config.BGL_DATA_PATH):
        raise FileNotFoundError(f"BGL data not found at {Config.BGL_DATA_PATH}")
    
    bgl_data = pd.read_parquet(Config.BGL_DATA_PATH)
    bgl_data = bgl_data[bgl_data['dt'] >= 0]  # Filter valid timestamps
    
    case_ids = bgl_data['caseid'].unique()
    if limit_cases:
        case_ids = case_ids[:limit_cases]
        print(f"Processing limited to first {limit_cases} cases")
    
    ppg_segments = []
    bgl_values = []
    
    # Process each case
    for case_id in tqdm(case_ids, desc="Processing PPG-BGL Alignment"):
        try:
            case_id_int = int(case_id)
            ppg_file_path = os.path.join(Config.PPG_DATA_DIR, f"{case_id_int}.npy")
            
            if not os.path.exists(ppg_file_path):
                continue
                
            # Load PPG signal
            ppg_signal = np.load(ppg_file_path)
            case_bgl_data = bgl_data[bgl_data['caseid'] == case_id_int]
            
            for _, row in case_bgl_data.iterrows():
                bgl_value = row['glucose']
                
                # Check if PPG signal is long enough for 10-second segment
                if len(ppg_signal) >= Config.SEGMENT_LENGTH:
                    # Extract last 10 seconds of data
                    segment = ppg_signal[-Config.SEGMENT_LENGTH:]
                    
                    # Apply Butterworth filter
                    try:
                        segment = apply_butterworth_filter(
                            segment, 
                            Config.LOWCUT, 
                            Config.HIGHCUT, 
                            Config.SAMPLING_RATE, 
                            Config.FILTER_ORDER
                        )
                    except:
                        # Skip if filtering fails
                        continue
                    
                    # Apply Min-Max normalization per signal
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    segment = scaler.fit_transform(segment.reshape(-1, 1)).flatten()
                    
                    # Store segment and BGL value
                    ppg_segments.append(segment)
                    bgl_values.append(bgl_value)
                    
        except Exception as e:
            print(f"❌ Error processing case {case_id}: {e}")
    
    # Convert to arrays
    ppg_segments = np.array(ppg_segments)
    bgl_values = np.array(bgl_values)
    
    print(f"✅ Loaded {len(ppg_segments)} PPG segments")
    return ppg_segments, bgl_values

def split_and_filter_data(ppg_segments, bgl_values):
    """
    Split data into train/test and filter by BGL range
    Following the stratified splitting approach from the notebook
    """
    print(f"\n🔄 Splitting and filtering data...")
    
    train_ppg = []
    train_bgl = []
    test_ppg = []
    test_bgl = []
    
    # Process data in BGL ranges (stratified splitting)
    for low in range(50, 150, 10):
        high = low + 10
        indices_in_range = np.where((bgl_values >= low) & (bgl_values < high))[0]
        
        if len(indices_in_range) > 1:
            # Shuffle and split in half
            random.shuffle(indices_in_range)
            split_index = len(indices_in_range) // 2
            train_indices = indices_in_range[:split_index]
            test_indices = indices_in_range[split_index:]
        else:
            # Not enough data to split
            train_indices = indices_in_range
            test_indices = []
        
        # Add to train/test sets
        if len(train_indices) > 0:
            train_ppg.extend(ppg_segments[train_indices])
            train_bgl.extend(bgl_values[train_indices])
        if len(test_indices) > 0:
            test_ppg.extend(ppg_segments[test_indices])
            test_bgl.extend(bgl_values[test_indices])
    
    # Convert to arrays
    train_ppg = np.array(train_ppg)
    train_bgl = np.array(train_bgl)
    test_ppg = np.array(test_ppg)
    test_bgl = np.array(test_bgl)
    
    print(f"✅ Training data: {len(train_bgl)} samples")
    print(f"✅ Testing data: {len(test_bgl)} samples")
    
    return train_ppg, train_bgl, test_ppg, test_bgl

def augment_training_data(train_ppg, train_bgl):
    """
    Apply targeted Gaussian noise augmentation to training data
    Focus on underrepresented BGL ranges (< 80, especially 40-60)
    """
    print(f"\n🔄 Applying targeted Gaussian noise augmentation...")
    
    augmented_ppg = []
    augmented_bgl = []
    
    for segment, bgl_value in tqdm(zip(train_ppg, train_bgl), desc="Augmenting Data", total=len(train_ppg)):
        
        # General augmentation for BGL < 80
        if bgl_value < 80:
            for noise_level in Config.NOISE_LEVELS_GENERAL:
                noise = np.random.normal(0, noise_level, segment.shape)
                augmented_segment = segment + noise
                augmented_segment = np.clip(augmented_segment, 0, 1)  # Clip to [0,1]
                
                augmented_ppg.append(augmented_segment)
                augmented_bgl.append(bgl_value)
        
        # Extra targeted augmentation for very low BGL (40-60)
        if 40 <= bgl_value <= 60:
            for _ in range(Config.EXTRA_AUGMENTATIONS):
                for noise_level in Config.NOISE_LEVELS_TARGETED:
                    noise = np.random.normal(0, noise_level, segment.shape)
                    augmented_segment = segment + noise
                    augmented_segment = np.clip(augmented_segment, 0, 1)
                    
                    augmented_ppg.append(augmented_segment)
                    augmented_bgl.append(bgl_value)
    
    # Combine original and augmented data
    combined_ppg = np.concatenate([train_ppg, np.array(augmented_ppg)], axis=0)
    combined_bgl = np.concatenate([train_bgl, np.array(augmented_bgl)], axis=0)
    
    print(f"✅ Original training samples: {len(train_ppg)}")
    print(f"✅ Augmented samples added: {len(augmented_ppg)}")
    print(f"✅ Total training samples: {len(combined_ppg)}")
    
    return combined_ppg, combined_bgl

def downsample_by_range(ppg_segments, bgl_values):
    """
    Downsample data to balance BGL ranges (max samples per 10 mg/dL range)
    """
    print(f"\n🔄 Downsampling to balance BGL ranges...")
    
    # Group data by BGL ranges
    range_groups = {}
    for i, bgl_value in enumerate(bgl_values):
        range_key = f"{int(bgl_value // 10) * 10}-{int(bgl_value // 10) * 10 + 10}"
        if range_key not in range_groups:
            range_groups[range_key] = []
        range_groups[range_key].append(i)
    
    # Downsample each range
    downsampled_indices = []
    for range_key, indices in range_groups.items():
        if len(indices) > Config.MAX_SAMPLES_PER_RANGE:
            indices = random.sample(indices, Config.MAX_SAMPLES_PER_RANGE)
        downsampled_indices.extend(indices)
        print(f"{range_key}: {len(indices)} samples after downsampling")
    
    # Extract downsampled data
    downsampled_ppg = ppg_segments[downsampled_indices]
    downsampled_bgl = bgl_values[downsampled_indices]
    
    print(f"✅ Total samples after downsampling: {len(downsampled_bgl)}")
    return downsampled_ppg, downsampled_bgl

def create_data_loaders(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    """Create PyTorch data loaders"""
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)
    
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
    """Train the CNN-GRU hybrid model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.L1Loss()  # MAE loss for regression
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n🚀 Training CNN-GRU hybrid model...")
    
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
        
        with torch.no_grad():
            for X_val_batch, Y_val_batch in val_loader:
                X_val_batch, Y_val_batch = X_val_batch.to(device), Y_val_batch.to(device)
                val_outputs = model(X_val_batch)
                val_loss += criterion(val_outputs, Y_val_batch).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(Config.RESULTS_DIR, "models", "best_cnn_gru_model.pth")
            torch.save(model.state_dict(), model_path)
        
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

def evaluate_model(model, test_loader):
    """Evaluate model performance on test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            true_values.extend(Y_batch.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    # Calculate regression metrics
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    r2 = r2_score(true_values, predictions)
    
    results = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'predictions': predictions.tolist(),
        'true_values': true_values.tolist()
    }
    
    print(f"\n📊 Experiment 2 Test Results (Regression):")
    print(f"Mean Absolute Error (MAE): {mae:.4f} mg/dL")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} mg/dL")
    print(f"R² Score: {r2:.4f}")
    
    return results

def plot_training_history(history):
    """Plot and save training history"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(history['train_losses'], label='Training Loss (MAE)')
    ax.plot(history['val_losses'], label='Validation Loss (MAE)')
    ax.set_title('Experiment 2 - Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MAE)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(Config.RESULTS_DIR, "plots", "training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Training plot saved: {plot_path}")

def plot_predictions(predictions, true_values):
    """Plot prediction vs true values scatter plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot: Predictions vs True Values
    ax1.scatter(true_values, predictions, alpha=0.6)
    ax1.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 
             color='red', linestyle='--', label='Perfect Prediction')
    ax1.set_title('Experiment 2 - Predicted vs True BGL')
    ax1.set_xlabel('True Blood Glucose (mg/dL)')
    ax1.set_ylabel('Predicted Blood Glucose (mg/dL)')
    ax1.legend()
    ax1.grid(True)
    
    # Error distribution histogram
    errors = true_values - predictions
    ax2.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_title('Prediction Error Distribution')
    ax2.set_xlabel('Prediction Error (True - Predicted)')
    ax2.set_ylabel('Frequency')
    ax2.axvline(0, color='red', linestyle='--', label='Zero Error')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(Config.RESULTS_DIR, "plots", "predictions_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Predictions plot saved: {plot_path}")

def plot_data_distribution(bgl_values, title_suffix=""):
    """Plot BGL data distribution"""
    plt.figure(figsize=(12, 6))
    plt.hist(bgl_values, bins=range(50, 200, 10), alpha=0.7, edgecolor='black')
    plt.title(f'BGL Distribution {title_suffix}')
    plt.xlabel('Blood Glucose Level (mg/dL)')
    plt.ylabel('Frequency')
    plt.xticks(range(50, 200, 10))
    plt.grid(True)
    
    plot_path = os.path.join(Config.RESULTS_DIR, "plots", f"bgl_distribution{title_suffix.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def main():
    """Main experiment runner"""
    print("🧪 EXPERIMENT 2: Augmented MUST Dataset with CNN-GRU Model")
    print("=" * 70)
    
    # Setup
    setup_results_directory()
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    random.seed(Config.RANDOM_SEED)
    
    start_time = datetime.now()
    
    try:
        # For development/testing, you can limit the number of cases
        limit_cases = None  # Set to e.g., 100 for testing
        
        # 1. Load and preprocess data
        ppg_segments, bgl_values = load_and_preprocess_data(limit_cases)
        
        if len(ppg_segments) == 0:
            raise ValueError("No data loaded")
        
        # Save original data distribution plot
        plot_data_distribution(bgl_values, "- Original Data")
        
        # 2. Split data into train/test (BEFORE augmentation)
        train_ppg, train_bgl, test_ppg, test_bgl = split_and_filter_data(ppg_segments, bgl_values)
        
        # 3. Apply targeted augmentation to training data only
        augmented_ppg, augmented_bgl = augment_training_data(train_ppg, train_bgl)
        
        # Save augmented data distribution plot
        plot_data_distribution(augmented_bgl, "- After Augmentation")
        
        # 4. Downsample to balance ranges
        final_ppg, final_bgl = downsample_by_range(augmented_ppg, augmented_bgl)
        
        # Save final data distribution plot
        plot_data_distribution(final_bgl, "- Final Training Data")
        
        # 5. Create train/validation split from processed training data
        X_train, X_val, Y_train, Y_val = train_test_split(
            final_ppg, final_bgl, test_size=0.2, random_state=Config.RANDOM_SEED
        )
        
        # Filter test data to same BGL range
        test_mask = (test_bgl >= Config.BGL_MIN) & (test_bgl <= Config.BGL_MAX)
        X_test = test_ppg[test_mask]
        Y_test = test_bgl[test_mask]
        
        # Save processed data
        data_path = os.path.join(Config.RESULTS_DIR, "data", "processed_data.npz")
        np.savez(data_path, 
                 X_train=X_train, Y_train=Y_train,
                 X_val=X_val, Y_val=Y_val,
                 X_test=X_test, Y_test=Y_test)
        print(f"💾 Processed data saved: {data_path}")
        
        # 6. Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, X_val, X_test, Y_train, Y_val, Y_test
        )
        
        # 7. Initialize model
        model = CNNGRUHybrid(
            gru_hidden_size=Config.GRU_HIDDEN_SIZE,
            gru_layers=Config.GRU_LAYERS,
            dropout=Config.DROPOUT
        )
        print(f"🏗️  CNN-GRU model initialized")
        
        # 8. Train model
        history = train_model(model, train_loader, val_loader)
        
        # 9. Load best model for evaluation
        best_model_path = os.path.join(Config.RESULTS_DIR, "models", "best_cnn_gru_model.pth")
        model.load_state_dict(torch.load(best_model_path))
        
        # 10. Evaluate on test set
        results = evaluate_model(model, test_loader)
        
        # 11. Generate plots
        plot_training_history(history)
        plot_predictions(np.array(results['predictions']), np.array(results['true_values']))
        
        # 12. Save final model (matching expected filename)
        final_model_path = os.path.join(Config.RESULTS_DIR, "models", "cnn_gru_hybrid_model.pth")
        torch.save(model.state_dict(), final_model_path)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 13. Compile experiment info
        experiment_info = {
            'experiment_name': 'Experiment 2: Augmented MUST Dataset with CNN-GRU Model',
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'config': {
                'sampling_rate': Config.SAMPLING_RATE,
                'segment_length': Config.SEGMENT_LENGTH,
                'bgl_range': [Config.BGL_MIN, Config.BGL_MAX],
                'noise_levels_general': Config.NOISE_LEVELS_GENERAL,
                'noise_levels_targeted': Config.NOISE_LEVELS_TARGETED,
                'extra_augmentations': Config.EXTRA_AUGMENTATIONS,
                'max_samples_per_range': Config.MAX_SAMPLES_PER_RANGE,
                'batch_size': Config.BATCH_SIZE,
                'epochs': Config.EPOCHS,
                'learning_rate': Config.LEARNING_RATE,
                'gru_hidden_size': Config.GRU_HIDDEN_SIZE,
                'gru_layers': Config.GRU_LAYERS,
                'dropout': Config.DROPOUT,
                'filter_params': {
                    'order': Config.FILTER_ORDER,
                    'lowcut': Config.LOWCUT,
                    'highcut': Config.HIGHCUT
                }
            },
            'data_info': {
                'original_samples': int(len(ppg_segments)),
                'train_samples_final': int(len(X_train)),
                'val_samples': int(len(X_val)),
                'test_samples': int(len(X_test)),
                'augmentation_strategy': 'targeted_gaussian_noise_after_split',
                'model_type': 'cnn_gru_hybrid_regression'
            },
            'training_history': history,
            'test_results': results,
            'files': {
                'processed_data': data_path,
                'best_model': best_model_path,
                'final_model': final_model_path,
                'training_plot': os.path.join(Config.RESULTS_DIR, "plots", "training_history.png"),
                'predictions_plot': os.path.join(Config.RESULTS_DIR, "plots", "predictions_analysis.png")
            },
            'key_findings': {
                'model_architecture': 'dual_cnn_pathway_plus_bidirectional_gru',
                'augmentation_approach': 'targeted_gaussian_noise_for_underrepresented_bgl',
                'data_integrity': 'split_before_augmentation_prevents_leakage',
                'test_mae': results['mae'],
                'test_rmse': results['rmse'],
                'test_r2': results['r2_score']
            }
        }
        
        # 14. Log experiment
        log_experiment_info(experiment_info)
        
        print(f"\n✅ Experiment 2 completed successfully!")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"📁 Results saved in: {Config.RESULTS_DIR}")
        
        return experiment_info
        
    except Exception as e:
        print(f"❌ Error in Experiment 2: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()