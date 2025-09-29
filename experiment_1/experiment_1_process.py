#!/usr/bin/env python3
"""
Experiment 1: Window Selection with CNN-LSTM and ResNet - Preprocessing
====================================================================

This script implements the preprocessing pipeline for Experiment 1:
1. Bandpass filtering (0.5-8 Hz) of raw PPG signals
2. Extract last 8 (or 16) 10-second segments before each BGL reading
3. Select best 1-second window from each segment using cosine similarity
4. Normalize windows independently to reduce inter-subject variability

Final output: (N_samples, 8, 100) for 8 windows of 1-second data
"""

import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import butter, filtfilt

# Configuration
class Config:
    # Data parameters
    SAMPLING_RATE = 100      # Hz
    WINDOW_SIZE = 100        # 1-second window (100 samples)
    SEGMENT_LENGTH = 10      # 10-second segments
    NUM_SEGMENTS = 8         # Extract last 8 segments (80 seconds)
    # NUM_SEGMENTS = 16      # Alternative: 16 segments (160 seconds)
    
    # Filter parameters (Bandpass 0.5-8 Hz)
    FILTER_ORDER = 3
    LOWCUT = 0.5   # Hz
    HIGHCUT = 8.0  # Hz
    
    # Paths
    BGL_DATA_PATH = "cleaned_bgl_data.parquet"
    PPG_DATA_DIR = "output/ppg_filtered_v3/"
    OUTPUT_DIR = "output/experiment_1_data/"
    
def setup_output_directory():
    """Create output directory structure"""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

def apply_bandpass_filter(signal, lowcut, highcut, fs, order=3):
    """Apply Butterworth bandpass filter (0.5-8 Hz)"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Ensure signal is long enough for filtering
    if len(signal) > max(len(b), 21):
        return filtfilt(b, a, signal)
    else:
        print(f"Signal too short ({len(signal)} samples), skipping filter")
        return signal

def score_window_quality(window, fs=100):
    """
    Score window quality based on:
    - Peak prominence
    - Signal smoothness
    - Amplitude variation
    """
    try:
        # Detect peaks in window
        peak_dict = nk.ppg_findpeaks(window, sampling_rate=fs)
        peaks = peak_dict["PPG_Peaks"]
        
        # Must have exactly one peak
        if len(peaks) != 1:
            return -np.inf
        
        peak_idx = peaks[0]
        
        # Peak prominence: difference between peak and surrounding valleys
        left_valley = np.min(window[:peak_idx]) if peak_idx > 0 else window[peak_idx]
        right_valley = np.min(window[peak_idx:]) if peak_idx < len(window)-1 else window[peak_idx]
        peak_prominence = window[peak_idx] - max(left_valley, right_valley)
        
        # Signal smoothness (lower std = smoother)
        signal_smoothness = -np.std(window)
        
        # Amplitude variation
        amplitude_range = np.mean(np.abs(np.diff(window)))
        
        # Combined score
        score = (peak_prominence * 1.5) + (signal_smoothness * 0.5) + (amplitude_range * 0.2)
        return score
        
    except:
        return -np.inf

def extract_best_windows_from_segment(segment, fs=100):
    """
    Extract all 1-second windows from a 10-second segment and return the best one
    based on quality scoring
    """
    window_size = Config.WINDOW_SIZE
    
    # Detect peaks in the segment
    try:
        peak_dict = nk.ppg_findpeaks(segment, sampling_rate=fs)
        peaks = peak_dict["PPG_Peaks"]
    except:
        return None
    
    if len(peaks) == 0:
        return None
    
    best_window = None
    best_score = -np.inf
    
    # Extract windows around each peak
    for peak_idx in peaks:
        start_w = peak_idx - (window_size // 2)
        end_w = peak_idx + (window_size // 2)
        
        # Skip if window goes out of bounds
        if start_w < 0 or end_w > len(segment):
            continue
            
        window = segment[start_w:end_w]
        
        # Score this window
        score = score_window_quality(window, fs)
        
        if score > best_score:
            best_score = score
            best_window = window
    
    return best_window

def compute_template_waveform():
    """
    Compute template waveform by averaging all extracted 1-second windows
    This template will be used for cosine similarity selection
    """
    print("🔄 Computing template waveform from all extracted windows...")
    
    bgl_data = pd.read_parquet(Config.BGL_DATA_PATH)
    all_windows = []
    
    # Collect windows from all cases
    for case_id in tqdm(bgl_data['caseid'].unique()[:100], desc="Collecting windows for template"):  # Limit for speed
        try:
            case_id_int = int(case_id)
            ppg_file_path = os.path.join(Config.PPG_DATA_DIR, f"{case_id_int}.npy")
            
            if not os.path.exists(ppg_file_path):
                continue
                
            ppg_signal = np.load(ppg_file_path)
            case_bgl = bgl_data[bgl_data['caseid'] == case_id]
            
            for _, row in case_bgl.iterrows():
                timestamp = row['dt']
                
                # Extract PPG window
                start_idx = max(0, int((timestamp - (Config.NUM_SEGMENTS * Config.SEGMENT_LENGTH)) * Config.SAMPLING_RATE))
                end_idx = int(timestamp * Config.SAMPLING_RATE)
                
                if end_idx - start_idx != (Config.NUM_SEGMENTS * Config.SEGMENT_LENGTH * Config.SAMPLING_RATE):
                    continue
                    
                ppg_window = ppg_signal[start_idx:end_idx]
                
                # Apply bandpass filter
                ppg_window = apply_bandpass_filter(
                    ppg_window, Config.LOWCUT, Config.HIGHCUT, Config.SAMPLING_RATE, Config.FILTER_ORDER
                )
                
                # Divide into segments
                segments = [ppg_window[i * Config.SAMPLING_RATE * Config.SEGMENT_LENGTH:(i + 1) * Config.SAMPLING_RATE * Config.SEGMENT_LENGTH] 
                           for i in range(Config.NUM_SEGMENTS)]
                
                # Extract windows from each segment
                for segment in segments:
                    if len(segment) == 0:
                        continue
                        
                    # Get all valid windows from this segment
                    try:
                        peak_dict = nk.ppg_findpeaks(segment, sampling_rate=Config.SAMPLING_RATE)
                        peaks = peak_dict["PPG_Peaks"]
                        
                        for peak_idx in peaks:
                            start_w = peak_idx - (Config.WINDOW_SIZE // 2)
                            end_w = peak_idx + (Config.WINDOW_SIZE // 2)
                            
                            if start_w >= 0 and end_w <= len(segment):
                                window = segment[start_w:end_w]
                                all_windows.append(window)
                    except:
                        continue
                        
        except Exception as e:
            print(f"❌ Error processing case {case_id}: {e}")
    
    if len(all_windows) == 0:
        raise ValueError("No windows extracted for template computation")
    
    # Compute mean template
    template = np.mean(all_windows, axis=0)
    
    # Save template
    template_path = os.path.join(Config.OUTPUT_DIR, "template_waveform.npy")
    np.save(template_path, template)
    
    print(f"✅ Template computed from {len(all_windows)} windows")
    print(f"💾 Template saved: {template_path}")
    
    return template

def select_best_window_cosine_similarity(segment_windows, template):
    """
    Select the best window from a segment using cosine similarity to template
    """
    if len(segment_windows) == 0:
        return None
    
    scaler = StandardScaler()
    best_window = None
    best_similarity = -1
    
    # Standardize template for comparison
    template_standardized = scaler.fit_transform(template.reshape(-1, 1)).flatten()
    
    for window in segment_windows:
        # Standardize window
        window_standardized = scaler.fit_transform(window.reshape(-1, 1)).flatten()
        
        # Compute cosine similarity
        similarity = cosine_similarity(
            window_standardized.reshape(1, -1),
            template_standardized.reshape(1, -1)
        )[0][0]
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_window = window_standardized  # Return standardized window
    
    return best_window

def process_experiment_1_data():
    """
    Main processing function for Experiment 1
    
    Steps:
    1. Load BGL data and PPG signals
    2. Extract last 8 (or 16) 10-second segments before each BGL
    3. Apply bandpass filtering (0.5-8 Hz)
    4. Select best 1-second window from each segment using cosine similarity
    5. Normalize windows independently
    """
    print("🧪 EXPERIMENT 1: Window Selection with CNN-LSTM and ResNet - Data Processing")
    print("=" * 80)
    
    setup_output_directory()
    
    # Load template waveform (compute if doesn't exist)
    template_path = os.path.join(Config.OUTPUT_DIR, "template_waveform.npy")
    if os.path.exists(template_path):
        template = np.load(template_path)
        print(f"📁 Loaded existing template: {template_path}")
    else:
        template = compute_template_waveform()
    
    # Load BGL data
    if not os.path.exists(Config.BGL_DATA_PATH):
        raise FileNotFoundError(f"BGL data not found at {Config.BGL_DATA_PATH}")
    
    bgl_data = pd.read_parquet(Config.BGL_DATA_PATH)
    
    # Initialize data storage
    X_data, Y_data = [], []
    
    print(f"\n🔄 Processing PPG-BGL alignment...")
    print(f"📊 Configuration: {Config.NUM_SEGMENTS} segments × {Config.SEGMENT_LENGTH}s = {Config.NUM_SEGMENTS * Config.SEGMENT_LENGTH}s total")
    
    # Process each case
    for case_id in tqdm(bgl_data['caseid'].unique(), desc="Processing cases"):
        try:
            case_id_int = int(case_id)
            ppg_file_path = os.path.join(Config.PPG_DATA_DIR, f"{case_id_int}.npy")
            
            if not os.path.exists(ppg_file_path):
                continue
                
            ppg_signal = np.load(ppg_file_path)
            case_bgl = bgl_data[bgl_data['caseid'] == case_id]
            
            for _, row in case_bgl.iterrows():
                timestamp, glucose = row['dt'], row['glucose']
                
                # Extract PPG window (last N*10 seconds before BGL)
                start_idx = max(0, int((timestamp - (Config.NUM_SEGMENTS * Config.SEGMENT_LENGTH)) * Config.SAMPLING_RATE))
                end_idx = int(timestamp * Config.SAMPLING_RATE)
                
                if end_idx - start_idx != (Config.NUM_SEGMENTS * Config.SEGMENT_LENGTH * Config.SAMPLING_RATE):
                    continue
                    
                ppg_window = ppg_signal[start_idx:end_idx]
                
                # Apply bandpass filter (0.5-8 Hz)
                ppg_window = apply_bandpass_filter(
                    ppg_window, Config.LOWCUT, Config.HIGHCUT, Config.SAMPLING_RATE, Config.FILTER_ORDER
                )
                
                # Divide into segments
                segments = [ppg_window[i * Config.SAMPLING_RATE * Config.SEGMENT_LENGTH:(i + 1) * Config.SAMPLING_RATE * Config.SEGMENT_LENGTH] 
                           for i in range(Config.NUM_SEGMENTS)]
                
                selected_windows = []
                
                for segment in segments:
                    if len(segment) == 0:
                        continue
                    
                    # Extract all candidate windows from segment
                    try:
                        peak_dict = nk.ppg_findpeaks(segment, sampling_rate=Config.SAMPLING_RATE)
                        peaks = peak_dict["PPG_Peaks"]
                        
                        if len(peaks) == 0:
                            continue
                        
                        segment_windows = []
                        for peak_idx in peaks:
                            start_w = peak_idx - (Config.WINDOW_SIZE // 2)
                            end_w = peak_idx + (Config.WINDOW_SIZE // 2)
                            
                            if start_w >= 0 and end_w <= len(segment):
                                window = segment[start_w:end_w]
                                
                                # Check for single peak
                                try:
                                    window_peaks = nk.ppg_findpeaks(window, sampling_rate=Config.SAMPLING_RATE)["PPG_Peaks"]
                                    if len(window_peaks) == 1:
                                        segment_windows.append(window)
                                except:
                                    continue
                        
                        # Select best window using cosine similarity
                        if len(segment_windows) > 0:
                            best_window = select_best_window_cosine_similarity(segment_windows, template)
                            if best_window is not None:
                                selected_windows.append(best_window)
                                
                    except Exception as e:
                        continue
                
                # Only keep samples with exactly the required number of segments
                if len(selected_windows) == Config.NUM_SEGMENTS:
                    X_data.append(np.array(selected_windows))
                    Y_data.append(glucose)
                    
        except Exception as e:
            print(f"❌ Error processing case {case_id}: {e}")
    
    # Convert to arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    
    print(f"\n✅ Processing completed!")
    print(f"📊 Final dataset shape:")
    print(f"   - X_data: {X_data.shape} (Samples, Segments, Window)")
    print(f"   - Y_data: {Y_data.shape} (Glucose values)")
    
    # Save processed data
    x_path = os.path.join(Config.OUTPUT_DIR, f"X_train_{Config.NUM_SEGMENTS}_segments.npy")
    y_path = os.path.join(Config.OUTPUT_DIR, f"Y_train_{Config.NUM_SEGMENTS}_segments.npy")
    
    np.save(x_path, X_data)
    np.save(y_path, Y_data)
    
    print(f"💾 Data saved:")
    print(f"   - X_data: {x_path}")
    print(f"   - Y_data: {y_path}")
    
    return X_data, Y_data

def main():
    """Main function"""
    try:
        X_data, Y_data = process_experiment_1_data()
        
        print(f"\n📈 Data Statistics:")
        print(f"   - Glucose range: [{np.min(Y_data):.1f}, {np.max(Y_data):.1f}] mg/dL")
        print(f"   - Glucose mean: {np.mean(Y_data):.1f} ± {np.std(Y_data):.1f} mg/dL")
        print(f"   - PPG window range: [{np.min(X_data):.3f}, {np.max(X_data):.3f}]")
        
        print(f"\n🎯 Ready for CNN-LSTM and ResNet training!")
        print(f"📁 All files saved in: {Config.OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()