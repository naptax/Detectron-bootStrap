"""
Script pour évaluer un modèle Detectron2 de segmentation sémantique sur un dataset COCO (validation)

Exemple d'utilisation :
    python eval_detectron2_semseg.py --config output_semseg/config.yaml --weights output_semseg/model_final.pth --val-json ./dataset/valid/annotations.coco.json --val-img-dir ./dataset/valid

"""
import argparse
import os

parser = argparse.ArgumentParser(description="Évaluation d'un modèle Detectron2 sur un dataset COCO.")
parser.add_argument('--config', type=str, required=True, help='Chemin du fichier config.yaml généré à l\'entraînement')
parser.add_argument('--weights', type=str, required=True, help='Chemin du fichier de poids .pth')
parser.add_argument('--val-json', type=str, required=True, help='Fichier COCO annotations validation')
parser.add_argument('--val-img-dir', type=str, required=True, help='Dossier images validation')
parser.add_argument('--output', type=str, default='eval_output', help='Répertoire de sortie pour les résultats')
args = parser.parse_args()

import torch
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os

# Enregistrement du dataset de validation
register_coco_instances("my_val_eval", {}, args.val_json, args.val_img_dir)

# Chargement de la config et des poids
cfg = get_cfg()
cfg.merge_from_file(args.config)
cfg.MODEL.WEIGHTS = args.weights
cfg.DATASETS.TEST = ("my_val_eval",)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.output, exist_ok=True)

# Évaluation COCO
evaluator = COCOEvaluator("my_val_eval", output_dir=args.output)
val_loader = build_detection_test_loader(cfg, "my_val_eval")

print("\n\033[1;34m========== [Début de l'évaluation] ==========" + "\033[0m")
results = inference_on_dataset(DefaultPredictor(cfg).model, val_loader, evaluator)
print("\033[1;32m[Résultats COCO]\033[0m")
print(results)
print(f"\nLes métriques COCO sont sauvegardées dans : {args.output}")
