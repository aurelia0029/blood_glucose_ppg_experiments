#!/usr/bin/env python3
"""
Experiment 1: Modified ResNet34 Model for Blood Glucose Prediction
=================================================================

Modified ResNet34 Architecture for 1D Time-Series:
- Conv1d (8→64) → BatchNorm → ReLU
- 4 stages of residual blocks (3, 4, 6, 3 blocks respectively) 
- Skip connections with dropout
- Global Average Pooling over time dimension
- Dense layers: 512 → 256 → 128 → 1 with ReLU + Dropout(0.3)

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
    INITIAL_CHANNELS = 64    # Initial conv layer channels
    DROPOUT = 0.3            # Dropout rate
    
    # ResNet stages configuration
    RESNET_LAYERS = [3, 4, 6, 3]  # Number of blocks in each stage
    RESNET_CHANNELS = [64, 128, 256, 512]  # Channels in each stage
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.001     # L2 regularization
    
    # Paths
    DATA_DIR = "output/experiment_1_data/"
    RESULTS_DIR = "experiments/experiment_1_resnet_results/"
    
    # Random seed
    RANDOM_SEED = 42

def setup_results_directory():
    """Create results directory structure"""
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.RESULTS_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(Config.RESULTS_DIR, "plots"), exist_ok=True)

def log_experiment_info(info_dict):
    """Log experiment information to JSON file"""
    log_file = os.path.join(Config.RESULTS_DIR, "experiment_1_resnet_log.json")
    with open(log_file, 'w') as f:
        json.dump(info_dict, f, indent=2)
    print(f"✅ Experiment log saved to: {log_file}")

class ResidualBlock1D(nn.Module):
    """
    1D Residual Block for time-series data
    
    Supports both identity and projection shortcuts
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super(ResidualBlock1D, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Skip connection
        out = self.relu(out)
        
        return out

class ResNet1D(nn.Module):
    """
    Modified ResNet34 for 1D Time-Series Blood Glucose Prediction
    
    Architecture:
    - Initial conv layer (8 → 64 channels)
    - 4 stages of residual blocks
    - Global average pooling
    - Fully connected layers for regression
    """
    def __init__(self, 
                 num_segments=8,
                 window_size=100, 
                 initial_channels=64,
                 resnet_layers=[3, 4, 6, 3],
                 resnet_channels=[64, 128, 256, 512],
                 dropout=0.3):
        super(ResNet1D, self).__init__()
        
        self.num_segments = num_segments
        self.window_size = window_size
        self.dropout = dropout
        
        # Initial convolution layer
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels=num_segments, out_channels=initial_channels, 
                     kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(initial_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet stages
        self.stage1 = self._make_stage(initial_channels, resnet_channels[0], resnet_layers[0], stride=1)
        self.stage2 = self._make_stage(resnet_channels[0], resnet_channels[1], resnet_layers[1], stride=2)
        self.stage3 = self._make_stage(resnet_channels[1], resnet_channels[2], resnet_layers[2], stride=2)
        self.stage4 = self._make_stage(resnet_channels[2], resnet_channels[3], resnet_layers[3], stride=2)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers for regression
        self.fc = nn.Sequential(
            nn.Linear(resnet_channels[3], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(128, 1)  # Single output for glucose prediction
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """Create a ResNet stage with multiple residual blocks"""
        layers = []
        
        # First block may have stride > 1 for downsampling
        layers.append(ResidualBlock1D(in_channels, out_channels, stride, self.dropout))
        
        # Remaining blocks have stride = 1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, 1, self.dropout))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: (batch_size, num_segments, window_size)
        # Transpose to (batch_size, window_size, num_segments) for conv1d processing
        x = x.transpose(1, 2)  # (batch_size, window_size, num_segments)
        x = x.transpose(1, 2)  # Back to (batch_size, num_segments, window_size)
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # ResNet stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (batch_size, channels, 1)
        x = x.view(x.size(0), -1)    # Flatten to (batch_size, channels)
        
        # Fully connected layers
        glucose_pred = self.fc(x)
        
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
    """Train the ResNet34 model"""
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
    
    print(f"\n🚀 Training ResNet34 model...")
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
            model_path = os.path.join(Config.RESULTS_DIR, "models", "best_resnet34_model.pth")
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
    
    print(f"\n📊 ResNet34 Test Results:")
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
    plt.title('ResNet34 Training History')
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
    ax1.set_title('ResNet34: Predicted vs True Glucose')
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
    print("🧪 EXPERIMENT 1: ResNet34 Model for Blood Glucose Prediction")
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
        model = ResNet1D(
            num_segments=Config.NUM_SEGMENTS,
            window_size=Config.WINDOW_SIZE,
            initial_channels=Config.INITIAL_CHANNELS,
            resnet_layers=Config.RESNET_LAYERS,
            resnet_channels=Config.RESNET_CHANNELS,
            dropout=Config.DROPOUT
        )
        
        print(f"🏗️  ResNet34 model initialized")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        history = train_model(model, train_loader, val_loader)
        
        # Load best model for evaluation
        best_model_path = os.path.join(Config.RESULTS_DIR, "models", "best_resnet34_model.pth")
        model.load_state_dict(torch.load(best_model_path))
        
        # Evaluate model
        results = evaluate_model(model, test_loader, y_scaler)
        
        # Generate plots
        plot_training_history(history)
        plot_predictions(np.array(results['predictions']), np.array(results['true_values']))
        
        # Save final model
        final_model_path = os.path.join(Config.RESULTS_DIR, "models", "resnet34_final.pth")
        torch.save(model.state_dict(), final_model_path)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile experiment info
        experiment_info = {
            'experiment_name': 'Experiment 1: ResNet34 for Blood Glucose Prediction',
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'model_type': 'Modified ResNet34 for 1D Time-Series',
            'config': {
                'num_segments': Config.NUM_SEGMENTS,
                'window_size': Config.WINDOW_SIZE,
                'initial_channels': Config.INITIAL_CHANNELS,
                'resnet_layers': Config.RESNET_LAYERS,
                'resnet_channels': Config.RESNET_CHANNELS,
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
        
        print(f"\n✅ Experiment 1 (ResNet34) completed successfully!")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"📁 Results saved in: {Config.RESULTS_DIR}")
        
    except Exception as e:
        print(f"❌ Error in Experiment 1 (ResNet34): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()