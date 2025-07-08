"""
Script pour vérifier les images manquantes référencées dans un fichier COCO (annotations.coco.json)

Utilisation :
    python check_missing_images.py --json ./dataset/train/annotations.coco.json --images ./dataset/train/images
"""
import os
import json
import argparse

parser = argparse.ArgumentParser(description="Vérifie les images référencées dans un COCO JSON.")
parser.add_argument("--json", type=str, required=True, help="Chemin du fichier COCO annotations (ex: annotations.coco.json)")
parser.add_argument("--images", type=str, required=True, help="Dossier contenant les images")
args = parser.parse_args()

with open(args.json, 'r') as f:
    coco = json.load(f)

image_files = set(os.listdir(args.images))
referenced_files = set(img['file_name'] for img in coco['images'])

missing = [f for f in referenced_files if f not in image_files]

if missing:
    print(f"\033[1;31m{len(missing)} fichier(s) image référencé(s) dans le JSON sont absents du dossier images :\033[0m")
    for f in missing:
        print(f"  - {f}")
else:
    print("\033[1;32mToutes les images référencées sont présentes dans le dossier.\033[0m")
