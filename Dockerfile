FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENV DISPLAY=:99
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    xvfb \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN pip3 install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Create and activate virtual environment, install dependencies
RUN python3.11 -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv sync --all-extras && \
    uv pip install vllm flash-attention && \
    playwright install chromium && \
    playwright install-deps

# Create comprehensive Streamlit config
RUN mkdir -p /root/.streamlit && \
    echo '[browser]\n\
gatherUsageStats = false\n\
serverAddress = "0.0.0.0"\n\
serverPort = 8501\n\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
[theme]\n\
base = "dark"\n\
[logger]\n\
level = "error"' > /root/.streamlit/config.toml

# Create .streamlit in home directory as well
RUN mkdir -p /.streamlit && \
    cp /root/.streamlit/config.toml /.streamlit/config.toml

# Create directory for model mounting
RUN mkdir -p /models

# Expose ports for Streamlit and vLLM
EXPOSE 8501
EXPOSE 8008

# Create entrypoint script
RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1280x1024x24 & \n\
sleep 1\n\
source /app/.venv/bin/activate\n\
\n\
# Start vLLM server in background without loading model\n\
vllm serve \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 8008 \
    --gpu-memory-utilization 0.6 \
    --max-num-batched-tokens 8192 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --block-size 16 \
    --max-num-seqs 32 \
    --enforce-eager & \n\
\n\
# Start Streamlit\n\
streamlit run --server.headless=true --server.address=0.0.0.0 --server.port=8501 src/proxy_lite/app.py' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"] 