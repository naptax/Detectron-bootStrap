"""
Script pour entraîner un modèle de segmentation sémantique avec Detectron2
-------------------------------------------------------------------------------
- Les données d'entraînement sont au format COCO et localisées dans dataset/train
- Les données de validation sont au format COCO et localisées dans dataset/valid
- Le backbone Detectron2 est sélectionnable via l'argument --backbone

Exemple d'utilisation :
    python train_detectron2_semseg.py --backbone mask_rcnn_R_50_FPN_3x

-------------------------------------------------------------------------------
"""

import sys
import argparse
import os

# Liste des backbones de segmentation sémantique courants (COCO-PanopticSegmentation)
BACKBONES = {
    "mask_rcnn_R_50_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "mask_rcnn_R_101_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "panoptic_fpn_R_50_3x": "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml",
    "panoptic_fpn_R_101_3x": "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
}

def print_backbones():
    print("\n\033[1;36mBackbones Detectron2 disponibles :\033[0m")
    for k, v in BACKBONES.items():
        print(f"  - \033[1m{k}\033[0m : {v}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*torch.meshgrid.*indexing argument.*")

    parser = argparse.ArgumentParser(description="Entraînement segmentation sémantique Detectron2.")
    parser.add_argument("-b", "--backbone", type=str, default="mask_rcnn_R_50_FPN_3x",
                        help="Nom du backbone à utiliser (voir liste dans le script)")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques d'entraînement")
    parser.add_argument("--output", type=str, default="output_semseg", help="Répertoire de sortie")
    parser.add_argument("--train-json", type=str, default="dataset/train/annotations.json", help="Chemin du fichier COCO annotations d'entraînement")
    parser.add_argument("--train-img-dir", type=str, default="dataset/train/images", help="Dossier des images d'entraînement")
    parser.add_argument("--val-json", type=str, default="dataset/valid/annotations.json", help="Chemin du fichier COCO annotations de validation")
    parser.add_argument("--val-img-dir", type=str, default="dataset/valid/images", help="Dossier des images de validation")
    args = parser.parse_args()

    if args.backbone not in BACKBONES:
        print("\033[1;31m[ERREUR] Le backbone demandé n'est pas supporté.\033[0m")
        print_backbones()
        sys.exit(3)

    print_backbones()
    print(f"\n\033[1;35mBackbone sélectionné :\033[0m \033[1m{args.backbone}\033[0m")

    # Imports Detectron2 et torch
    import detectron2
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog
    import torch

    # Enregistrement des datasets COCO
    train_json = args.train_json
    train_img_dir = args.train_img_dir
    val_json = args.val_json
    val_img_dir = args.val_img_dir

    register_coco_instances("my_train", {}, train_json, train_img_dir)
    register_coco_instances("my_val", {}, val_json, val_img_dir)

    # Extraction du nombre de classes à partir du fichier COCO
    import json
    try:
        with open(train_json, 'r') as f:
            coco_dict = json.load(f)
        categories = coco_dict.get('categories', [])
        if not categories:
            print(f"\033[1;31m[ERREUR] Pas de catégories trouvées dans le fichier {train_json}.\033[0m")
            sys.exit(4)
        num_classes = len(categories)
        print(f"\033[1;36mNombre de classes détectées : {num_classes}\033[0m")
    except Exception as e:
        print(f"\033[1;31m[ERREUR] Impossible de lire les catégories du COCO train ({train_json}) : {e}\033[0m")
        sys.exit(5)

    # Configuration Detectron2
    cfg = get_cfg()
    config_path = BACKBONES[args.backbone]
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.DATASETS.TRAIN = ("my_train",)
    cfg.DATASETS.TEST = ("my_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = args.output
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = args.epochs * 500  # à ajuster selon la taille du dataset
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Sauvegarde de la configuration complète dans le dossier de sortie
    config_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(config_path, "w") as f:
        f.write(cfg.dump())
    print(f"\033[1;36mConfiguration sauvegardée dans : {config_path}\033[0m")

    print(f"\n\033[1;34m========== [Début de l'entraînement] ==========" + "\033[0m")
    print(f"  Données train : {train_json} | {train_img_dir}")
    print(f"  Données val   : {val_json} | {val_img_dir}")
    print(f"  Résultats et modèles dans : {cfg.OUTPUT_DIR}")
    print(f"  Nombre d'époques : {args.epochs}")

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("\n\033[1;32m[Fin de l'entraînement]\033[0m")
    print(f"Les modèles et logs sont disponibles dans : {cfg.OUTPUT_DIR}")
