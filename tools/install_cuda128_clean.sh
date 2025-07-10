#!/bin/bash
# Script d'installation propre de CUDA 12.8 sous Ubuntu/WSL2
# 1. Installe CUDA 12.8 via apt
# 2. Configure PATH, LD_LIBRARY_PATH, update-alternatives, symlinks
# Usage : bash install_cuda128_clean.sh

set -e

# 1. Installer le dépôt CUDA officiel si besoin
if ! grep -q 'cuda/repos/ubuntu' /etc/apt/sources.list.d/cuda.list 2>/dev/null; then
  echo '[INFO] Ajout du dépôt CUDA 12.x officiel...'
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | \
    sudo gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | \
    sudo tee /etc/apt/sources.list.d/cuda.list
fi

# 2. Installation du toolkit
sudo apt update
sudo apt install -y cuda-toolkit-12-8

# 3. Corriger symlink /usr/local/cuda
if [ -d /usr/local/cuda-12.8 ]; then
  sudo ln -sf /usr/local/cuda-12.8 /usr/local/cuda
fi

# 4. update-alternatives pour cuda et nvcc
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-12.8 100
sudo update-alternatives --install /usr/bin/nvcc nvcc /usr/local/cuda-12.8/bin/nvcc 100
sudo update-alternatives --set cuda /usr/local/cuda-12.8
sudo update-alternatives --set nvcc /usr/local/cuda-12.8/bin/nvcc

# 5. Corriger PATH et LD_LIBRARY_PATH dans ~/.bashrc
BASHRC=~/.bashrc
if ! grep -q '/usr/local/cuda/bin' "$BASHRC"; then
    echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> "$BASHRC"
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> "$BASHRC"
    echo '[INFO] PATH et LD_LIBRARY_PATH corrigés dans ~/.bashrc'
fi

# 6. Recharger le shell
source "$BASHRC"

# 7. Vérification
ls -l /usr/local/cuda
ls -l /usr/bin/nvcc
nvcc --version
