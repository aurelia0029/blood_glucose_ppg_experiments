# Blood Glucose Prediction Experiments

This repository contains experiments 1-4 for blood glucose prediction using PPG signals.

## Available Experiments

### Experiment 1: Window Selection with CNN-LSTM and ResNet
- **Objective**: Compare CNN-LSTM vs ResNet34 for blood glucose classification
- **Key Features**:
  - Data preprocessing with bandpass filtering (0.5-8 Hz)
  - Template-based window selection using cosine similarity
  - Three-file architecture: preprocessing, CNN-LSTM, ResNet34
  - Modified ResNet34 for 1D time-series classification
  - Binary classification: Low (<70) vs Not Low (≥70) BGL

### Experiment 2: Augmented MUST Dataset with CNN-GRU Model
- **Objective**: Predict continuous BGL values using CNN-GRU hybrid regression model
- **Key Features**:
  - MUST dataset with 10-second PPG segments (1000 samples at 100 Hz)
  - Butterworth filtering (0.5-4 Hz) + Min-Max normalization per signal
  - Targeted Gaussian noise augmentation for underrepresented BGL ranges
  - Split BEFORE augmentation (prevents data leakage)
  - Dual CNN pathways + bidirectional GRU architecture
  - Regression output: Continuous BGL prediction

### Experiment 3: Feature-Enriched Classification Based on Peak-Centered PPG Windows
- **Objective**: Classify BGL using handcrafted features from peak-centered PPG windows
- **Key Features**: 
  - Peak-centered window extraction (10s before BGL to avoid data leakage)
  - Feature enrichment: HR, AUC, FW, FW_25, FW_50, FW_75
  - Data balancing AFTER train/test split
  - Test set maintains real-world class imbalance
  - Input format: 64 windows × 106 features (100 PPG + 6 extracted features)
  - Binary classification: Low (<70) vs Not Low (≥70) BGL

### Experiment 4: Classification with Augmentation Before Splitting
- **Objective**: Compare window timing strategies and data balancing order effects
- **Key Features**:
  - Data balancing BEFORE train/test split
  - **Variant A**: Windows immediately before BGL measurement (64×100 features)
  - **Variant B**: Windows 10s before BGL + feature extraction (64×106 features) 
  - Balanced test set with 2:1 ratio (Not Low:Low)
  - Side-by-side variant comparison
  - Binary classification: Low (<70) vs Not Low (≥70) BGL

## Quick Start

### Docker Setup (Recommended)

Each experiment has its own Docker container for better isolation and dependency management.

#### 1. Build Individual Experiment Images
```bash
# Build Experiment 1
cd blood_glucose_experiments/experiment_1
docker build -t blood-glucose-exp1 .

# Build Experiment 2
cd ../experiment_2
docker build -t blood-glucose-exp2 .

# Build Experiment 3
cd ../experiment_3
docker build -t blood-glucose-exp3 .

# Build Experiment 4
cd ../experiment_4
docker build -t blood-glucose-exp4 .
```

#### 2. Run Individual Experiments
```bash
# Run Experiment 1 (Window Selection with CNN-LSTM and ResNet)
# NOTE: Experiment 1 requires 3 sequential Docker commands
cd blood_glucose_experiments/experiment_1
docker run -v $(pwd)/results:/app/results blood-glucose-exp1 python3 experiment_1_process.py   # Step 1: Preprocessing
docker run -v $(pwd)/results:/app/results blood-glucose-exp1 python3 experiment_1_lstm.py     # Step 2: CNN-LSTM
docker run -v $(pwd)/results:/app/results blood-glucose-exp1 python3 experiment_1_resnet.py   # Step 3: ResNet34

# Run Experiment 2 (CNN-GRU Regression)
cd ../experiment_2
docker run -v $(pwd)/results:/app/results blood-glucose-exp2

# Run Experiment 3 (Feature-Enriched Classification)
cd ../experiment_3
docker run -v $(pwd)/results:/app/results blood-glucose-exp3

# Run Experiment 4 (Augmentation Before Splitting)
cd ../experiment_4
docker run -v $(pwd)/results:/app/results blood-glucose-exp4
```

#### 3. Interactive Mode (for debugging)
```bash
# Debug specific experiment
cd blood_glucose_experiments/experiment_X  # Replace X with experiment number
docker run -it -v $(pwd)/results:/app/results blood-glucose-expX bash
```

### Local Setup (Alternative)

#### 1. Install Dependencies for Specific Experiment
```bash
# For Experiment 1
cd blood_glucose_experiments/experiment_1
pip install -r requirements.txt

# For Experiment 2
cd ../experiment_2
pip install -r requirements.txt

# For Experiment 3
cd ../experiment_3
pip install -r requirements.txt

# For Experiment 4
cd ../experiment_4
pip install -r requirements.txt
```

#### 2. Run Experiments Locally
```bash
# Run Experiment 1 (3-step process)
cd blood_glucose_experiments/experiment_1
python3 experiment_1_process.py
python3 experiment_1_lstm.py
python3 experiment_1_resnet.py

# Run Experiment 2
cd ../experiment_2
python3 experiment_2.py

# Run Experiment 3
cd ../experiment_3
python3 experiment_3.py

# Run Experiment 4
cd ../experiment_4
python3 experiment_4.py
```

## Required Data Files
Ensure these files are present in each experiment directory:
- `cleaned_bgl_data.parquet` - Cleaned BGL data
- `output/ppg_filtered_v3/` - Directory containing PPG signal files (*.npy)

## Expected Output Structure

### Experiment 1 Results
```
experiment_1_results/
├── models/
│   ├── best_cnn_lstm_model.pth
│   ├── best_resnet_model.pth
│   ├── final_cnn_lstm_model.pth
│   └── final_resnet_model.pth
├── plots/
│   ├── training_history_lstm.png
│   ├── training_history_resnet.png
│   ├── confusion_matrix_lstm.png
│   └── confusion_matrix_resnet.png
├── data/
│   ├── processed_data.npz
│   └── selected_windows.npz
├── experiment_1_lstm_log.json
├── experiment_1_resnet_log.json
└── model_comparison.json
```

### Experiment 2 Results
```
experiment_2_results/
├── models/
│   ├── best_cnn_gru_model.pth
│   └── cnn_gru_hybrid_model.pth
├── plots/
│   ├── training_history.png
│   ├── predictions_analysis.png
│   ├── bgl_distribution_-_original_data.png
│   ├── bgl_distribution_-_after_augmentation.png
│   └── bgl_distribution_-_final_training_data.png
├── data/
│   └── processed_data.npz
└── experiment_2_log.json
```

### Experiment 3 Results
```
experiment_3_results/
├── models/
│   ├── best_cnn_model.pth
│   └── cnn_binary_classification.pth
├── plots/
│   ├── training_history.png
│   └── confusion_matrix.png
├── data/
│   └── raw_data.npz
└── experiment_3_log.json
```

### Experiment 4 Results
```
experiment_4_results/
├── models/
│   ├── best_model_variant_A.pth
│   ├── best_model_variant_B.pth
│   ├── final_model_variant_A.pth
│   └── final_model_variant_B.pth
├── plots/
│   ├── training_history_variant_A.png
│   ├── training_history_variant_B.png
│   ├── confusion_matrix_variant_A.png
│   └── confusion_matrix_variant_B.png
├── data/
│   ├── raw_data_variant_A.npz
│   └── raw_data_variant_B.npz
├── experiment_log_variant_A.json
├── experiment_log_variant_B.json
└── variant_comparison.json
```

## Configuration

### Development Mode
For faster testing, you can limit the number of cases processed by modifying the `limit_cases` parameter in the `main()` function of either experiment:
```python
limit_cases = 100  # Process only first 100 cases instead of all
```

### Custom Parameters
Edit the `Config` class in each experiment file to modify:
- Data parameters (window size, number of windows, thresholds)
- Training parameters (batch size, epochs, learning rate)
- File paths and output directories

## Results Interpretation

### Experiment 2 Key Metrics (Regression)
- **Training History**: MAE loss curves for CNN-GRU hybrid model
- **Regression Metrics**: MAE, RMSE, R² score for continuous BGL prediction
- **Prediction Analysis**: Scatter plots of predicted vs true BGL values
- **Data Distribution**: Visualization of augmentation effects on BGL ranges
- **Model Architecture**: Dual CNN + bidirectional GRU feature fusion

### Experiment 3 Key Metrics (Classification)
- **Training History**: Loss and accuracy curves for balanced training data
- **Test Results**: Performance on unbalanced real-world distribution test set
- **Feature Impact**: Analysis of HR, AUC, FW features contribution
- **Real-World Performance**: How well the model performs with actual class imbalance
- **Peak-Centered Approach**: Effect of using heartbeat-centered window extraction

### Experiment 4 Key Metrics (Classification)
- **Variant Comparison**: Side-by-side comparison of timing strategies
- **Balancing Impact**: Effect of balancing before vs after splitting
- **Window Timing**: Immediate vs 10-second offset window extraction
- **Feature Engineering**: Raw PPG vs feature-enriched approaches
- **Data Leakage Prevention**: Impact of different augmentation timing

## Experiment Comparison Matrix

| Aspect | Experiment 1 | Experiment 2 | Experiment 3 | Experiment 4 |
|--------|-------------|-------------|-------------|-------------|
| **Task Type** | Binary Classification | Regression | Binary Classification | Binary Classification |
| **Model** | CNN-LSTM vs ResNet34 | CNN-GRU Hybrid | CNN | CNN |
| **Input** | Window-selected PPG | 10s PPG (1000 samples) | 64 windows × 106 features | 64 windows × 100/106 features |
| **Selection Method** | Cosine Similarity | Direct Segmentation | Peak-Centered | Immediate/10s Offset |
| **Augmentation** | None | Gaussian Noise | Oversampling/Undersampling | Oversampling/Undersampling |
| **Data Split** | Standard | Before Augmentation | After Split | Before Split |
| **Test Set** | Balanced | Balanced | Real-world Imbalance | Balanced |
| **Output** | Low/Not Low | Continuous BGL | Low/Not Low | Low/Not Low |
| **Metrics** | Accuracy, ROC AUC, F1 | MAE, RMSE, R² | Accuracy, ROC AUC, F1 | Accuracy, ROC AUC, F1 |

### Common Outputs
- **Training History**: Loss curves (MAE for regression, Cross-entropy for classification)
- **Model Files**: Best and final trained models
- **JSON Logs**: Detailed experiment configuration and results
- **Data Files**: Processed datasets for reproducibility

## Architecture Benefits of Individual Docker Containers

### Isolation and Dependency Management
- Each experiment has its own Python environment with specific dependencies
- Prevents version conflicts between different experiment requirements
- Easier maintenance and updates for individual experiments

### Resource Management
- Run only the experiment you need without loading unnecessary dependencies
- Better memory management for large-scale experiments
- Parallel execution of different experiments on different machines

### Development Workflow
- Independent development and testing of each experiment
- Easier debugging with focused container environments
- Version control of experiment-specific configurations

## Hardware Requirements
- **CPU**: Multi-core recommended for faster processing
- **Memory**: 8GB+ RAM recommended
- **GPU**: Optional (CUDA support available)
- **Storage**: 5GB+ for data and results

## Troubleshooting

### Common Issues
1. **Missing Data Files**: Ensure `cleaned_bgl_data.parquet` and PPG files exist
2. **Memory Issues**: Reduce batch size or limit number of cases
3. **CUDA Errors**: Set `CUDA_VISIBLE_DEVICES=""` to disable GPU

### Docker Issues
```bash
# Check container logs
docker logs <container_id>

# Debug interactively
docker run -it blood-glucose-experiments bash
```