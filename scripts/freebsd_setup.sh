# [[file:../SETUP.org::*FreeBSD System Dependencies][FreeBSD System Dependencies:1]]
#!/bin/sh

# FreeBSD 14.3 system setup for LNN development
echo "Setting up FreeBSD environment for Liquid Neural Networks..."

# Install base development tools
sudo pkg install -y \
    git \
    curl \
    wget \
    bash \
    zsh \
    tmux \
    emacs \
    vim \
    python39 \
    python39-pip \
    openjdk11 \
    leiningen \
    clojure \
    rust \
    cargo \
    julia \
    gcc \
    cmake \
    pkgconf \
    libxml2 \
    libxslt \
    sqlite3 \
    postgresql13-client \
    redis \
    graphviz \
    gnuplot \
    imagemagick7 \
    ffmpeg \
    pandoc \
    texlive-base \
    texlive-latex-extra

# Install scientific computing libraries
sudo pkg install -y \
    py39-numpy \
    py39-scipy \
    py39-matplotlib \
    py39-pandas \
    py39-scikit-learn \
    py39-jupyter \
    py39-notebook \
    blas \
    lapack \
    openblas \
    fftw3 \
    gsl \
    hdf5

# Install ML/AI specific tools
sudo pkg install -y \
    py39-torch \
    py39-torchvision \
    py39-tensorflow \
    py39-keras \
    py39-statsmodels \
    py39-sympy \
    py39-networkx

echo "✓ FreeBSD base system configured"
# FreeBSD System Dependencies:1 ends here
