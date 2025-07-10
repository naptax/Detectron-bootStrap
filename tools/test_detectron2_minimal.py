"""
Script Python minimal pour valider l'installation et le fonctionnement de Detectron2
-------------------------------------------------------------------------------
Ce script réalise :
  1. L'import de Detectron2, l'affichage de sa version et du chemin du module.
  2. Le chargement d'un modèle pré-entraîné (Faster R-CNN, COCO) via model_zoo.
  3. Un test de prédiction sur une image factice (aléatoire) avec le predictor.
  4. L'affichage des clés de sortie du modèle pour valider le fonctionnement.

Utilisation :
    python test_detectron2_minimal.py
    python test_detectron2_minimal.py --backbone mask_rcnn_R_101_FPN_3x

Sortie attendue :
    - Detectron2 importé avec succès
    - Détection réussie. Clés de sortie : ['instances']
    - Test Detectron2 : OK

Si une erreur survient ou si Detectron2 n'est pas installé, un message explicite s'affiche.
-------------------------------------------------------------------------------
"""

import sys
import argparse

# Liste des backbones Detectron2 disponibles (COCO-Detection)
BACKBONES = {
    "faster_rcnn_R_50_FPN_3x": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "faster_rcnn_R_101_FPN_3x": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "faster_rcnn_X_101_32x8d_FPN_3x": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "faster_rcnn_R_50_DC5_3x": "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml",
    "faster_rcnn_R_50_C4_3x": "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
    "retinanet_R_50_FPN_3x": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    "retinanet_R_101_FPN_3x": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
    "mask_rcnn_R_50_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "mask_rcnn_R_101_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
}

def print_backbones():
    print("\n\033[1;36mBackbones Detectron2 disponibles :\033[0m")
    for k, v in BACKBONES.items():
        print(f"  - \033[1m{k}\033[0m : {v}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*torch.meshgrid.*indexing argument.*")

    parser = argparse.ArgumentParser(description="Test Detectron2 avec sélection du backbone.")
    parser.add_argument("-b", "--backbone", type=str, default="faster_rcnn_R_50_FPN_3x",
                        help="Nom du backbone à utiliser (voir liste dans le script)")
    args = parser.parse_args()

    if args.backbone not in BACKBONES:
        print("\033[1;31m[ERREUR] Le backbone demandé n'est pas supporté.\033[0m")
        print_backbones()
        sys.exit(3)

    print_backbones()
    print(f"\n\033[1;35mBackbone sélectionné :\033[0m \033[1m{args.backbone}\033[0m")

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
        import numpy as np

        # Configuration rapide avec un modèle COCO (Faster R-CNN)
        cfg = get_cfg()
        # Utilisation du backbone choisi
        config_path = BACKBONES[args.backbone]
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # seuil de détection
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
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

