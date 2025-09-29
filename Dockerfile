# Blood Glucose Prediction Experiments Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy experiment scripts
COPY experiment_*.py ./
COPY *.py ./

# Copy data files (adjust paths as needed)
COPY cleaned_bgl_data.parquet ./
COPY output/ ./output/

# Create results directory
RUN mkdir -p experiments

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""

# Default command (can be overridden)
CMD ["python3", "experiment_4.py"]

# Example usage:
# docker build -t blood-glucose-experiments .
# docker run -v $(pwd)/experiments:/app/experiments blood-glucose-experiments
# docker run -v $(pwd)/experiments:/app/experiments blood-glucose-experiments python3 experiment_1.py
# docker run -v $(pwd)/experiments:/app/experiments blood-glucose-experiments python3 experiment_2.py
# etc.