#!/usr/bin/env python3
"""
Experiment 1: CNN-LSTM Model for Blood Glucose Prediction
========================================================

CNN-LSTM Hybrid Architecture:
- CNN Feature Extractor: Conv1d (8→64) → Conv1d (64→128) → MaxPool → Dropout(0.3)
- LSTM: Two layers with input_size=128, hidden_size=128, dropout=0.3
- Output: Linear(128 → 1) for regression

Input: (N_samples, 8, 100) - 8 selected 1-second windows
Output: Continuous blood glucose values

Training Setup:
- Loss: MSE (Mean Squared Error)
- Optimizer: Adam with lr=0.0005, weight_decay=0.001
- Epochs: 30
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configuration
class Config:
    # Model parameters
    NUM_SEGMENTS = 8         # Number of 1-second windows
    WINDOW_SIZE = 100        # Samples per window
    CNN_CHANNELS_1 = 64      # First CNN layer output channels
    CNN_CHANNELS_2 = 128     # Second CNN layer output channels
    LSTM_HIDDEN_SIZE = 128   # LSTM hidden size
    LSTM_LAYERS = 2          # Number of LSTM layers
    DROPOUT = 0.3            # Dropout rate
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.001     # L2 regularization
    
    # Paths
    DATA_DIR = "output/experiment_1_data/"
    RESULTS_DIR = "experiments/experiment_1_lstm_results/"
    
    # Random seed
    RANDOM_SEED = 42

def setup_results_directory():
    """Create results directory structure"""
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.RESULTS_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(Config.RESULTS_DIR, "plots"), exist_ok=True)

def log_experiment_info(info_dict):
    """Log experiment information to JSON file"""
    log_file = os.path.join(Config.RESULTS_DIR, "experiment_1_lstm_log.json")
    with open(log_file, 'w') as f:
        json.dump(info_dict, f, indent=2)
    print(f"✅ Experiment log saved to: {log_file}")

class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM Hybrid Model for Blood Glucose Prediction
    
    Architecture:
    1. CNN Feature Extractor processes each 1-second window
    2. LSTM processes the sequence of CNN features across 8 windows
    3. Final linear layer outputs glucose prediction
    """
    def __init__(self, 
                 num_segments=8, 
                 window_size=100,
                 cnn_channels_1=64,
                 cnn_channels_2=128,
                 lstm_hidden_size=128,
                 lstm_layers=2,
                 dropout=0.3):
        super(CNNLSTMModel, self).__init__()
        
        self.num_segments = num_segments
        self.window_size = window_size
        
        # CNN Feature Extractor for each 1-second window
        self.cnn = nn.Sequential(
            # First Conv layer: 8 → 64 channels
            nn.Conv1d(in_channels=1, out_channels=cnn_channels_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels_1),
            
            # Second Conv layer: 64 → 128 channels  
            nn.Conv1d(in_channels=cnn_channels_1, out_channels=cnn_channels_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels_2),
            
            # Max pooling and dropout
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        
        # Calculate CNN output size
        # Input: 100 samples → After Conv layers and MaxPool: 100/2 = 50
        self.cnn_output_size = 50  # After maxpool with kernel_size=2
        
        # Global Average Pooling to reduce CNN output to fixed size
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # LSTM to process sequence of CNN features
        self.lstm = nn.LSTM(
            input_size=cnn_channels_2,  # 128 features from CNN
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Final regression layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Single output for glucose prediction
        )
        
    def forward(self, x):
        # Input shape: (batch_size, num_segments, window_size)
        batch_size, num_segments, window_size = x.shape
        
        # Process each window through CNN
        cnn_features = []
        for i in range(num_segments):
            # Extract one window: (batch_size, window_size)
            window = x[:, i, :].unsqueeze(1)  # Add channel dim: (batch_size, 1, window_size)
            
            # Pass through CNN
            cnn_out = self.cnn(window)  # (batch_size, 128, reduced_length)
            
            # Global average pooling to get fixed-size features
            pooled = self.global_avg_pool(cnn_out)  # (batch_size, 128, 1)
            pooled = pooled.squeeze(-1)  # (batch_size, 128)
            
            cnn_features.append(pooled)
        
        # Stack CNN features to create sequence
        # Shape: (batch_size, num_segments, 128)
        cnn_sequence = torch.stack(cnn_features, dim=1)
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(cnn_sequence)
        
        # Use the last LSTM output
        last_output = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)
        
        # Final prediction
        glucose_pred = self.fc(last_output)  # (batch_size, 1)
        
        return glucose_pred

def load_data():
    """Load preprocessed data from experiment_1_process.py"""
    x_path = os.path.join(Config.DATA_DIR, f"X_train_{Config.NUM_SEGMENTS}_segments.npy")
    y_path = os.path.join(Config.DATA_DIR, f"Y_train_{Config.NUM_SEGMENTS}_segments.npy")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Processed data not found. Please run experiment_1_process.py first.\n"
            f"Expected files:\n  - {x_path}\n  - {y_path}"
        )
    
    X_data = np.load(x_path)
    Y_data = np.load(y_path)
    
    print(f"✅ Loaded data:")
    print(f"   - X_data: {X_data.shape}")
    print(f"   - Y_data: {Y_data.shape}")
    
    return X_data, Y_data

def prepare_data(X_data, Y_data):
    """Prepare data for training"""
    # Train-validation-test split
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X_data, Y_data, test_size=0.2, random_state=Config.RANDOM_SEED
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, test_size=0.25, random_state=Config.RANDOM_SEED  # 0.25 * 0.8 = 0.2 overall
    )
    
    # Standardize glucose values
    y_scaler = StandardScaler()
    Y_train_scaled = y_scaler.fit_transform(Y_train.reshape(-1, 1)).flatten()
    Y_val_scaled = y_scaler.transform(Y_val.reshape(-1, 1)).flatten()
    Y_test_scaled = y_scaler.transform(Y_test.reshape(-1, 1)).flatten()
    
    print(f"📊 Data splits:")
    print(f"   - Training: {X_train.shape[0]} samples")
    print(f"   - Validation: {X_val.shape[0]} samples")
    print(f"   - Test: {X_test.shape[0]} samples")
    
    return (X_train, X_val, X_test, 
            Y_train_scaled, Y_val_scaled, Y_test_scaled, 
            y_scaler)

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
    """Train the CNN-LSTM model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n🚀 Training CNN-LSTM model...")
    print(f"🔧 Device: {device}")
    
    for epoch in range(Config.EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for X_batch, Y_batch in train_loader:
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
            model_path = os.path.join(Config.RESULTS_DIR, "models", "best_cnn_lstm_model.pth")
            torch.save(model.state_dict(), model_path)
        
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

def evaluate_model(model, test_loader, y_scaler):
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
    
    # Inverse transform to get original glucose values
    predictions_orig = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    true_values_orig = y_scaler.inverse_transform(true_values.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(true_values_orig, predictions_orig)
    mse = mean_squared_error(true_values_orig, predictions_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values_orig, predictions_orig)
    
    results = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'predictions': predictions_orig.tolist(),
        'true_values': true_values_orig.tolist()
    }
    
    print(f"\n📊 CNN-LSTM Test Results:")
    print(f"Mean Absolute Error (MAE): {mae:.4f} mg/dL")
    print(f"Mean Squared Error (MSE): {mse:.4f} mg/dL²")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} mg/dL")
    print(f"R² Score: {r2:.4f}")
    
    return results

def plot_training_history(history):
    """Plot and save training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Training Loss (MSE)')
    plt.plot(history['val_losses'], label='Validation Loss (MSE)')
    plt.title('CNN-LSTM Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(Config.RESULTS_DIR, "plots", "training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Training plot saved: {plot_path}")

def plot_predictions(predictions, true_values):
    """Plot prediction vs true values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(true_values, predictions, alpha=0.6)
    ax1.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 
             'r--', label='Perfect Prediction')
    ax1.set_xlabel('True Glucose (mg/dL)')
    ax1.set_ylabel('Predicted Glucose (mg/dL)')
    ax1.set_title('CNN-LSTM: Predicted vs True Glucose')
    ax1.legend()
    ax1.grid(True)
    
    # Error distribution
    errors = true_values - predictions
    ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Prediction Error (mg/dL)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Prediction Error Distribution')
    ax2.axvline(0, color='red', linestyle='--', label='Zero Error')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(Config.RESULTS_DIR, "plots", "predictions_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Predictions plot saved: {plot_path}")

def main():
    """Main experiment runner"""
    print("🧪 EXPERIMENT 1: CNN-LSTM Model for Blood Glucose Prediction")
    print("=" * 70)
    
    # Setup
    setup_results_directory()
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    start_time = datetime.now()
    
    try:
        # Load data
        X_data, Y_data = load_data()
        
        # Prepare data
        X_train, X_val, X_test, Y_train, Y_val, Y_test, y_scaler = prepare_data(X_data, Y_data)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, X_val, X_test, Y_train, Y_val, Y_test
        )
        
        # Initialize model
        model = CNNLSTMModel(
            num_segments=Config.NUM_SEGMENTS,
            window_size=Config.WINDOW_SIZE,
            cnn_channels_1=Config.CNN_CHANNELS_1,
            cnn_channels_2=Config.CNN_CHANNELS_2,
            lstm_hidden_size=Config.LSTM_HIDDEN_SIZE,
            lstm_layers=Config.LSTM_LAYERS,
            dropout=Config.DROPOUT
        )
        
        print(f"🏗️  CNN-LSTM model initialized")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        history = train_model(model, train_loader, val_loader)
        
        # Load best model for evaluation
        best_model_path = os.path.join(Config.RESULTS_DIR, "models", "best_cnn_lstm_model.pth")
        model.load_state_dict(torch.load(best_model_path))
        
        # Evaluate model
        results = evaluate_model(model, test_loader, y_scaler)
        
        # Generate plots
        plot_training_history(history)
        plot_predictions(np.array(results['predictions']), np.array(results['true_values']))
        
        # Save final model
        final_model_path = os.path.join(Config.RESULTS_DIR, "models", "cnn_lstm_final.pth")
        torch.save(model.state_dict(), final_model_path)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile experiment info
        experiment_info = {
            'experiment_name': 'Experiment 1: CNN-LSTM for Blood Glucose Prediction',
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'model_type': 'CNN-LSTM Hybrid',
            'config': {
                'num_segments': Config.NUM_SEGMENTS,
                'window_size': Config.WINDOW_SIZE,
                'cnn_channels': [Config.CNN_CHANNELS_1, Config.CNN_CHANNELS_2],
                'lstm_hidden_size': Config.LSTM_HIDDEN_SIZE,
                'lstm_layers': Config.LSTM_LAYERS,
                'dropout': Config.DROPOUT,
                'batch_size': Config.BATCH_SIZE,
                'epochs': Config.EPOCHS,
                'learning_rate': Config.LEARNING_RATE,
                'weight_decay': Config.WEIGHT_DECAY
            },
            'data_info': {
                'total_samples': int(len(X_data)),
                'train_samples': int(len(X_train)),
                'val_samples': int(len(X_val)),
                'test_samples': int(len(X_test)),
                'input_shape': list(X_data.shape[1:])
            },
            'training_history': history,
            'test_results': results,
            'files': {
                'best_model': best_model_path,
                'final_model': final_model_path,
                'training_plot': os.path.join(Config.RESULTS_DIR, "plots", "training_history.png"),
                'predictions_plot': os.path.join(Config.RESULTS_DIR, "plots", "predictions_analysis.png")
            }
        }
        
        # Log experiment
        log_experiment_info(experiment_info)
        
        print(f"\n✅ Experiment 1 (CNN-LSTM) completed successfully!")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"📁 Results saved in: {Config.RESULTS_DIR}")
        
    except Exception as e:
        print(f"❌ Error in Experiment 1 (CNN-LSTM): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()