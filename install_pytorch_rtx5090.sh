#!/bin/bash
# =====================================================================
# Script d'installation de PyTorch nightly compatible CUDA 12.8 (RTX 5090)
# Utilise un environnement Python isolé avec uv et venv
# Ce script :
#   - Installe uv si besoin
#   - Crée un venv Python .venv
#   - Installe PyTorch nightly + torchvision + torchaudio pour CUDA 12.8
#   - Vérifie l'installation et la compatibilité CUDA
# =====================================================================
set -e

# 1. Vérifier/installer uv
if ! command -v uv &> /dev/null; then
  echo "[INFO] Installation de uv..."
  pip install uv
fi

# 2. Créer un environnement virtuel Python
if [ ! -d .venv ]; then
  echo "[INFO] Création de l'environnement virtuel Python..."
  uv venv .venv
fi
source .venv/bin/activate

# 3. Installer PyTorch nightly compatible CUDA 12.8
# (torch, torchvision, torchaudio)
echo "[INFO] Installation de PyTorch nightly (CUDA 12.8) via uv..."
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. Vérification
python -c 'import torch; print("Torch version:", torch.__version__); print("CUDA dispo:", torch.cuda.is_available()); print("CUDA version:", torch.version.cuda)'

# 5. Afficher la version de nvcc pour vérification
nvcc --version

echo "[SUCCESS] PyTorch nightly (CUDA 12.8) installé et prêt pour RTX 5090 !"
