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
    print(f"Detectron2 importé avec succès. Version : {getattr(detectron2, '__version__', 'inconnue')}")
    print(f"Chemin du module : {getattr(detectron2, '__file__', 'inconnu')}")
except ImportError:
    print("[ERREUR] Detectron2 n'est pas installé ou importable !")
    exit(1)

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
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Création du predictor Detectron2
    predictor = DefaultPredictor(cfg)

    # Générer une image factice (RGB, 800x600)
    img = (np.random.rand(600,800,3)*255).astype(np.uint8)
    outputs = predictor(img)
    print("Détection réussie. Clés de sortie :", list(outputs.keys()))
    print("Test Detectron2 : OK")
except Exception as e:
    print(f"[ERREUR] Problème lors du test Detectron2 : {e}")
    exit(2)
