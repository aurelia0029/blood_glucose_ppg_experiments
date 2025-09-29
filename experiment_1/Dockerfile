# Experiment 1: Window Selection with CNN-LSTM and ResNet
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy experiment files
COPY experiment_1_*.py ./
COPY cleaned_bgl_data.parquet ./
COPY output/ ./output/

# Create output directories
RUN mkdir -p output/experiment_1_data experiments

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""

# Default command runs preprocessing
CMD ["python3", "experiment_1_process.py"]

# Usage examples:
# docker build -t experiment-1 .
# docker run -v $(pwd)/results:/app/experiments experiment-1 python3 experiment_1_process.py
# docker run -v $(pwd)/results:/app/experiments experiment-1 python3 experiment_1_lstm.py  
# docker run -v $(pwd)/results:/app/experiments experiment-1 python3 experiment_1_resnet.py