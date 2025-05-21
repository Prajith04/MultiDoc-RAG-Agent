# Dockerfile - Container configuration for the MultiDoc RAG Agent
#
# This Dockerfile sets up a containerized environment to run the RAG agent:
# 1. Uses Python 3.10 as the base image
# 2. Installs system dependencies and Python packages
# 3. Configures model caching and environment variables
# 4. Exposes the Gradio web interface port

# Use Python 3.10 slim as the base image for smaller size
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Create cache directory for model downloads
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

# Set environment variables for model caching and application configuration
ENV TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache \
    SENTENCE_TRANSFORMERS_HOME=/app/cache \
    PORT=7860 \
    PYTHONUNBUFFERED=1

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install gradio

# Expose Gradio port for web access
EXPOSE 7860

# Run the Gradio web application
CMD ["python", "gradio_demo.py"]
