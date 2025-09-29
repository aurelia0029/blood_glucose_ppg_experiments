# Experiment 4: Classification with Augmentation Before Splitting
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
COPY experiment_4.py ./
COPY cleaned_bgl_data.parquet ./
COPY output/ ./output/

# Create output directories
RUN mkdir -p experiments

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""

# Default command
CMD ["python3", "experiment_4.py"]

# Usage:
# docker build -t experiment-4 .
# docker run -v $(pwd)/results:/app/experiments experiment-4