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
    parser.add_argument("--max-iter", type=int, default=5000, help="Nombre d'itérations d'entraînement (Detectron2-style)")
    parser.add_argument("--output", type=str, default="output_semseg", help="Répertoire de sortie racine (les runs seront dans des sous-dossiers datés)")
    parser.add_argument("--train-json", type=str, default="dataset/train/annotations.json", help="Chemin du fichier COCO annotations d'entraînement")
    parser.add_argument("--train-img-dir", type=str, default="dataset/train/images", help="Dossier des images d'entraînement")
    parser.add_argument("--val-json", type=str, default="dataset/valid/annotations.json", help="Chemin du fichier COCO annotations de validation")
    parser.add_argument("--val-img-dir", type=str, default="dataset/valid/images", help="Dossier des images de validation")
    parser.add_argument("--augment", dest="augment", action="store_true", help="Activer la data augmentation avancée (par défaut)")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Désactiver la data augmentation avancée")
    parser.set_defaults(augment=True)
    args = parser.parse_args()

    from datetime import datetime
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.output, run_time)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\033[1;36mDossier de sortie du run : {run_dir}\033[0m")

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
    detectron2_config_path = BACKBONES[args.backbone]
    cfg.merge_from_file(model_zoo.get_config_file(detectron2_config_path))
    cfg.DATASETS.TRAIN = ("my_train",)
    cfg.DATASETS.TEST = ("my_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = run_dir
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = args.max_iter  # Detectron2-style
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Sauvegarde de la configuration complète dans le dossier de sortie
    output_config_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(output_config_path, "w") as f:
        f.write(cfg.dump())
    print(f"\033[1;36mConfiguration sauvegardée dans : {output_config_path}\033[0m")

    print(f"\n\033[1;34m========== [Début de l'entraînement] ==========" + "\033[0m")
    print(f"  Données train : {train_json} | {train_img_dir}")
    print(f"  Données val   : {val_json} | {val_img_dir}")
    print(f"  Résultats et modèles dans : {cfg.OUTPUT_DIR}")
    print(f"  Nombre d'itérations (max_iter Detectron2) : {args.max_iter}")

    import detectron2.data.transforms as T
    from detectron2.data import DatasetMapper, build_detection_train_loader
    from detectron2.engine import DefaultTrainer

    class CustomTrainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, cfg):
            if args.augment:
                augmentations = [
                    T.RandomRotation(angle=[90, 180, 270], sample_style="choice", expand=False),
                    T.RandomFlip(horizontal=True, vertical=True),
                    T.RandomBrightness(0.8, 1.2),
                    T.RandomContrast(0.8, 1.2),
                ]
                return build_detection_train_loader(
                    cfg,
                    mapper=DatasetMapper(cfg, is_train=True, augmentations=augmentations)
                )
            else:
                return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True))

    print(f"\033[1;34mData augmentation avancée : {'activée' if args.augment else 'désactivée'}\033[0m")
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # ===== Évaluation automatique à la fin de l'entraînement =====
    print("\n\033[1;34m========== [Évaluation COCO sur validation] ==========" + "\033[0m")
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    evaluator = COCOEvaluator("my_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "my_val")
    from detectron2.engine import DefaultPredictor
    predictor = DefaultPredictor(cfg)
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    import json
    metrics_path = os.path.join(cfg.OUTPUT_DIR, "metrics_coco.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\033[1;36mMétriques COCO sauvegardées dans : {metrics_path}\033[0m")

    # ===== Log CSV récapitulatif =====
    import csv
    log_path = os.path.join(args.output, "trainings_log.csv")
    log_exists = os.path.isfile(log_path)
    # Extraction des métriques principales
    def safe_get(d, *keys):
        for k in keys:
            d = d.get(k, {})
        return d if isinstance(d, (int, float)) else (d if d else None)
    row = {
        "run_time": run_time,
        "run_dir": run_dir,
        "backbone": args.backbone,

        "train_json": train_json,
        "val_json": val_json,
        "AP_bbox": safe_get(metrics, 'bbox', 'AP'),
        "AP50_bbox": safe_get(metrics, 'bbox', 'AP50'),
        "AP75_bbox": safe_get(metrics, 'bbox', 'AP75'),
        "AP_segm": safe_get(metrics, 'segm', 'AP'),
        "AP50_segm": safe_get(metrics, 'segm', 'AP50'),
        "AP75_segm": safe_get(metrics, 'segm', 'AP75'),
    }
    with open(log_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
        if not log_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"\033[1;36mLog mis à jour : {log_path}\033[0m")

    print("\n\033[1;32m[Fin de l'entraînement]\033[0m")
    print(f"Les modèles, logs et métriques sont disponibles dans : {cfg.OUTPUT_DIR}")
    print(f"Le tableau de suivi des runs est dans : {log_path}")
