# [[file:../SETUP.org::*Development Environment][Development Environment:1]]
#!/bin/sh

# Development environment setup
echo "Configuring development environment..."

# Python virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip setuptools wheel

# Core ML/AI libraries
pip install \
    torch torchvision torchaudio \
    tensorflow tensorflow-probability \
    jax jaxlib \
    numpy scipy matplotlib seaborn \
    pandas polars \
    scikit-learn scikit-image \
    plotly bokeh altair \
    jupyter jupyterlab \
    notebook nbconvert \
    ipywidgets \
    sympy \
    networkx \
    graph-tool \
    igraph \
    pyvis

# Specialized LNN libraries
pip install \
    diffrax \
    equinox \
    neural-tangents \
    dm-haiku \
    flax \
    optax \
    chex \
    jraph \
    orbax \
    blackjax

# Development tools
pip install \
    black isort flake8 mypy \
    pytest pytest-cov pytest-benchmark \
    hypothesis \
    pre-commit \
    sphinx sphinx-rtd-theme \
    mkdocs mkdocs-material \
    streamlit \
    gradio \
    wandb \
    tensorboard \
    mlflow

# Create .env file
cat > .env << 'EOF'
# Environment variables for LNN development
PYTHONPATH=src/python:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0
WANDB_PROJECT=liquid-neural-networks
MLFLOW_TRACKING_URI=http://localhost:5000
EOF

# Configure git hooks
pre-commit install

echo "✓ Development environment configured"
# Development Environment:1 ends here
