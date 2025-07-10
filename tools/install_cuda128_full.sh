#!/bin/bash
# =====================================================================
# Script d'installation propre de CUDA 12.8 avec headers et outils
# Pour Ubuntu 22.04 sous WSL2 - RTX 5090 (Ada Lovelace)
# Ce script :
#   - Désinstalle proprement toute version précédente de CUDA
#   - Ajoute le dépôt officiel NVIDIA CUDA
#   - Installe CUDA Toolkit 12.8 et tous les headers nécessaires
#   - Configure symlinks, update-alternatives, PATH et LD_LIBRARY_PATH
#   - Vérifie l'installation
# =====================================================================
set -e

# 1. Désinstaller toutes les anciennes versions de CUDA (optionnel mais recommandé)
echo "[INFO] Suppression des anciennes versions de CUDA..."
sudo apt-get remove --purge -y 'cuda*' 'nvidia-cuda*' 'libcudnn*' 'libnccl*' || true
sudo rm -rf /usr/local/cuda-*
sudo rm -rf /usr/local/cuda
sudo update-alternatives --remove-all cuda || true
sudo update-alternatives --remove-all nvcc || true
sudo apt-get autoremove -y
sudo apt-get autoclean -y

# 2. Ajouter le dépôt officiel NVIDIA CUDA
if ! grep -q 'cuda/repos/ubuntu' /etc/apt/sources.list.d/cuda.list 2>/dev/null; then
  echo '[INFO] Ajout du dépôt CUDA 12.x officiel...'
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | \
    sudo gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | \
    sudo tee /etc/apt/sources.list.d/cuda.list
fi

# 3. Installer CUDA Toolkit 12.8 et headers
sudo apt update
sudo apt install -y cuda-toolkit-12-8

# 4. Corriger symlink /usr/local/cuda
if [ -d /usr/local/cuda-12.8 ]; then
  sudo ln -sf /usr/local/cuda-12.8 /usr/local/cuda
fi

# 5. update-alternatives pour cuda et nvcc
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-12.8 100
sudo update-alternatives --install /usr/bin/nvcc nvcc /usr/local/cuda-12.8/bin/nvcc 100
sudo update-alternatives --set cuda /usr/local/cuda-12.8
sudo update-alternatives --set nvcc /usr/local/cuda-12.8/bin/nvcc

# 6. Corriger PATH et LD_LIBRARY_PATH dans ~/.bashrc
BASHRC=~/.bashrc
if ! grep -q '/usr/local/cuda/bin' "$BASHRC"; then
    echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> "$BASHRC"
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> "$BASHRC"
    echo '[INFO] PATH et LD_LIBRARY_PATH corrigés dans ~/.bashrc'
fi

# 7. Recharger le shell pour appliquer les variables d'environnement
source "$BASHRC"

# 8. Vérification
ls -l /usr/local/cuda
ls -l /usr/bin/nvcc
nvcc --version
ls /usr/local/cuda/include/cuda_runtime.h

echo "[SUCCESS] CUDA 12.8 installé et configuré !"
