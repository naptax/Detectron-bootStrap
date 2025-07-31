# Entraînement de segmentation sémantique avec Detectron2

[![Licence](https://img.shields.io/github/license/<OWNER>/<REPO>)](LICENSE)
[![Issues](https://img.shields.io/github/issues/<OWNER>/<REPO>)](https://github.com/<OWNER>/<REPO>/issues)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-compatible%20%7C%20cuda%2011%2B-important)](https://pytorch.org/)

---

Script minimal pour entraîner un modèle de segmentation sémantique avec [Detectron2](https://github.com/facebookresearch/detectron2) sur vos propres données COCO.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Ou installez Detectron2 selon votre version de CUDA :
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

Préparez vos données COCO dans :
- `dataset/train/images` et `dataset/train/annotations.json`
- `dataset/valid/images` et `dataset/valid/annotations.json`

## Lancement rapide

```bash
python train.py --backbone mask_rcnn_R_50_FPN_3x --max-iter 5000 \
  --train-json dataset/train/annotations.json --train-img-dir dataset/train/images \
  --val-json dataset/valid/annotations.json --val-img-dir dataset/valid/images
```

## Arguments principaux

| Argument                | Type    | Défaut                        | Description |
|-------------------------|---------|-------------------------------|-------------|
| `-b`, `--backbone`      | str     | `mask_rcnn_R_50_FPN_3x`       | Backbone Detectron2 à utiliser |
| `--max-iter`            | int     | `5000`                        | Nombre d'itérations d'entraînement |
| `--output`              | str     | `output_semseg`               | Dossier racine de sortie |
| `--train-json`          | str     | `dataset/train/annotations.json` | Fichier COCO d'entraînement |
| `--train-img-dir`       | str     | `dataset/train/images`         | Dossier images d'entraînement |
| `--val-json`            | str     | `dataset/valid/annotations.json` | Fichier COCO de validation |
| `--val-img-dir`         | str     | `dataset/valid/images`         | Dossier images de validation |
| `--augment` / `--no-augment` | bool | `--augment` (activé par défaut) | Active ou désactive la data augmentation avancée (rotations, flips, luminosité, contraste) pour l'entraînement |

Backbones disponibles : `mask_rcnn_R_50_FPN_3x`, `mask_rcnn_R_101_FPN_3x`, `panoptic_fpn_R_50_3x`, `panoptic_fpn_R_101_3x`

---

### Data augmentation avancée

- Par défaut, la data augmentation avancée est activée lors de l'entraînement :
    - Rotations multiples (90°, 180°, 270°)
    - Flips horizontaux et verticaux
    - Ajustements de luminosité et de contraste
- Pour désactiver ces transformations aléatoires, ajoutez simplement l'option `--no-augment` à votre commande.

**Exemples :**

- Avec data augmentation (par défaut) :
  ```bash
  python train.py --augment ...
  ```
  ou simplement
  ```bash
  python train.py ...
  ```
- Sans data augmentation :
  ```bash
  python train.py --no-augment ...
  ```

---

Ce projet est distribué sous licence MIT.


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
- Auteur principal : @NaPtaX
- Contributions bienvenues via Pull Request !
