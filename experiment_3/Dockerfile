# Experiment 3: Feature-Enriched Classification
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
COPY experiment_3.py ./
COPY cleaned_bgl_data.parquet ./
COPY output/ ./output/

# Create output directories
RUN mkdir -p experiments

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""

# Default command
CMD ["python3", "experiment_3.py"]

# Usage:
# docker build -t experiment-3 .
# docker run -v $(pwd)/results:/app/experiments experiment-3