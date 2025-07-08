"""
Script Python minimal pour valider l'installation et le fonctionnement de Detectron2

import warnings
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*indexing argument.*")
-------------------------------------------------------------------------------
Ce script réalise :
  1. L'import de Detectron2, l'affichage de sa version et du chemin du module.
  2. Le chargement d'un modèle pré-entraîné (Faster R-CNN, COCO) via model_zoo.
  3. Un test de prédiction sur une image factice (aléatoire) avec le predictor.
  4. L'affichage des clés de sortie du modèle pour valider le fonctionnement.

Utilisation :
    python test_detectron2_minimal.py

Sortie attendue :
    - Detectron2 importé avec succès
    - Détection réussie. Clés de sortie : ['instances']
    - Test Detectron2 : OK

Si une erreur survient ou si Detectron2 n'est pas installé, un message explicite s'affiche.
-------------------------------------------------------------------------------
"""

# 1. Test d'import Detectron2 et affichage version/chemin
try:
    import detectron2
    print("\n\033[1;34m========== [Detectron2] ==========" + "\033[0m")
    print(f"  Statut import : \033[1;32mOK\033[0m")
    print(f"  Version      : {getattr(detectron2, '__version__', 'inconnue')}")
    print(f"  Chemin       : {getattr(detectron2, '__file__', 'inconnu')}")
except ImportError:
    print("\033[1;31m[ERREUR] Detectron2 n'est pas installé ou importable !\033[0m")
    exit(1)

# Affichage des informations CUDA, NVIDIA, PyTorch et GPU
try:
    import torch
    print("\n\033[1;34m========== [PyTorch & CUDA] ==========" + "\033[0m")
    print(f"  PyTorch version    : {torch.__version__}")
    print(f"  CUDA disponible    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Version CUDA (PyT) : {torch.version.cuda}")
        print(f"  Nombre de GPU      : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    -> GPU {i} : {torch.cuda.get_device_name(i)}")
        print(f"  ID GPU courant     : {torch.cuda.current_device()}")
        try:
            # Cette méthode n'existe pas toujours selon la version de torch
            print(f"  Version driver NVIDIA : {torch._C._cuda_getDriverVersion()}")
        except Exception:
            print("  Version driver NVIDIA : inconnue (non supporté par torch)")
    else:
        print("  \033[1;33m[ALERTE] CUDA n'est pas disponible. Les traitements seront effectués sur le CPU !\033[0m")
except Exception as e:
    print(f"\033[1;31m[ERREUR] Problème lors de l'affichage des infos CUDA/PyTorch : {e}\033[0m")

# 2. Test d'un modèle pré-entraîné sur une image factice
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    import torch
    import numpy as np

    # Configuration rapide avec un modèle COCO (Faster R-CNN)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # seuil de détection
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    # On force l'utilisation du GPU si possible
    print("\n\033[1;34m========== [Configuration modèle Detectron2] ==========" + "\033[0m")
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        print("  \033[1;32m[INFO] Le modèle Detectron2 va utiliser le GPU (cuda).\033[0m")
    else:
        cfg.MODEL.DEVICE = "cpu"
        print("  \033[1;33m[ALERTE] Le modèle Detectron2 va utiliser le CPU (pas de GPU détecté).\033[0m")

    # Création du predictor Detectron2
    predictor = DefaultPredictor(cfg)

    # Générer une image factice (RGB, 800x600)
    img = (np.random.rand(600,800,3)*255).astype(np.uint8)
    outputs = predictor(img)
    print("\n\033[1;34m========== [Résultat prédiction] ==========" + "\033[0m")
    print("  Détection réussie. Clés de sortie :", list(outputs.keys()))
    print("  \033[1;32mTest Detectron2 : OK\033[0m")
except Exception as e:
    print(f"[ERREUR] Problème lors du test Detectron2 : {e}")
    exit(2)
