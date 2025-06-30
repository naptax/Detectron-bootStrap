#!/bin/bash
# =====================================================================
# Script d'installation de Detectron2 pour RTX 5090 (Ada, CUDA 12.8)
# Ce script suppose que CUDA 12.8 et PyTorch sont déjà installés et configurés
# Il NE touche PAS à CUDA ni à la version de PyTorch
# Il :
#   - Exporte explicitement l'environnement CUDA pour la compilation
#   - Installe les dépendances système nécessaires
#   - Clone Detectron2 si besoin
#   - Compile Detectron2 avec CUDA_ARCH=8.9 (Ada)
#   - Installe Detectron2 en mode editable dans l'environnement Python courant
# =====================================================================
set -e

# 1. S'assurer que l'environnement CUDA est bien exposé
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "[DEBUG] PATH=$PATH"
echo "[DEBUG] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# 2. Installer les dépendances système requises
sudo apt update
sudo apt install -y build-essential ninja-build libglib2.0-0 libsm6 libxrender1 libxext6 python3-dev

# 3. Cloner Detectron2 si besoin
if [ ! -d detectron2 ]; then
  echo "[INFO] Clonage du repo Detectron2..."
  git clone https://github.com/facebookresearch/detectron2.git
fi
cd detectron2

echo "[INFO] Nettoyage des anciens builds..."
python setup.py clean --all || true

# 4. Définir l'architecture CUDA pour Ada (RTX 5090)
export CUDA_ARCH=8.9
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"

# 5. Installer Detectron2 en mode editable
uv pip install -e . --no-build-isolation

# 6. Vérification
python -c 'import detectron2; print("Detectron2 installé dans:", detectron2.__file__)'

echo "[SUCCESS] Detectron2 compilé et installé pour RTX 5090 + CUDA 12.8 !"
