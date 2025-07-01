# Diagnostic GPU, CUDA et PyTorch sous WSL2

[![Build Status](https://img.shields.io/github/actions/workflow/status/<OWNER>/<REPO>/ci.yml?branch=main)](https://github.com/<OWNER>/<REPO>/actions)
[![Licence](https://img.shields.io/github/license/<OWNER>/<REPO>)](LICENSE)
[![Issues](https://img.shields.io/github/issues/<OWNER>/<REPO>)](https://github.com/<OWNER>/<REPO>/issues)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-nightly%20%7C%20cuda%2012.8-important)](https://pytorch.org/)

> Script de diagnostic et documentation pour installer et valider une chaîne GPU/CUDA/PyTorch compatible RTX 5090 sous WSL2 (Windows Subsystem for Linux 2).

---

## Sommaire
- [Présentation](#présentation)
- [Installation rapide](#installation-rapide)
- [Utilisation](#utilisation)
- [Documentation détaillée](#documentation-détaillée)
- [Licence](#licence)
- [Contributeurs](#contributeurs)

---

## Présentation
Ce projet fournit :
- Un script Python pour diagnostiquer la compatibilité GPU/CUDA/PyTorch, spécialement pour les GPU Ada Lovelace (RTX 5090).
- Une documentation complète pour installer CUDA et PyTorch sous WSL2.

## Installation rapide

1. **Créer et activer un environnement virtuel avec uv**
   ```bash
   uv venv
   source .venv/bin/activate
   ```
2. **Installer PyTorch (nightly, CUDA 12.8, RTX 5090)**
   ```bash
   uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```
3. **Installer le CUDA Toolkit dans WSL2**
   - Suivre le guide détaillé : [INSTALL_CUDA_WSL2.md](./INSTALL_CUDA_WSL2.md)

## Utilisation
Lancez le diagnostic complet avec :
```bash
python main.py
```

Vous obtiendrez un rapport sur :
- Le GPU détecté et ses caractéristiques
- La version CUDA disponible
- La version et la compatibilité PyTorch/CUDA
- Les conseils pour corriger la chaîne si besoin

## Documentation détaillée
Consultez le guide complet d'installation et de résolution de problèmes ici :
- [INSTALL_CUDA_WSL2.md](./INSTALL_CUDA_WSL2.md)

## Licence
Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](./LICENSE).

## Contributeurs
- Auteur principal : @<VOTRE_GITHUB>
- Contributions bienvenues via Pull Request !
