import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Chemins des fichiers
COCO_PATH = "dataset/valid/annotations.coco.json"
IMG_DIR = "dataset/valid/"

# Charger les annotations COCO
with open(COCO_PATH, 'r') as f:
    coco = json.load(f)

# Construire des index pour accès rapide
images = {img['id']: img for img in coco['images']}
categories = {cat['id']: cat for cat in coco['categories']}

# Grouper les annotations par image
annotations_by_image = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(ann)

def show_image_with_annotations(img_id):
    img_info = images[img_id]
    img_path = os.path.join(IMG_DIR, img_info['file_name'])
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image non trouvée : {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    patches = []
    colors = []
    anns = annotations_by_image.get(img_id, [])
    for ann in anns:
        cat = categories[ann['category_id']]['name']
        # Dessiner la bounding box
        bbox = ann['bbox']
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        # Afficher le nom de la catégorie
        ax.text(bbox[0], bbox[1] - 2, cat, fontsize=10, color='red', backgroundcolor='white')
        # Dessiner la segmentation
        segs = ann['segmentation']
        for seg in segs:
            poly = np.array(seg).reshape((-1, 2))
            polygon = Polygon(poly, closed=True)
            polygon.set_alpha(0.4)
            ax.add_patch(polygon)
    ax.axis('off')
    plt.title(f"Image: {img_info['file_name']} (id={img_id})")
    # Sauvegarde dans un dossier outputs/visualisations
    output_dir = os.path.join("outputs", "visualisations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(img_info['file_name']))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Image annotée sauvegardée : {output_path}")

# Exemple d'utilisation : afficher toutes les images annotées
def main():
    for img_id in images:
        show_image_with_annotations(img_id)
        # Pour tester sur une seule image, décommentez la ligne suivante
        # break

if __name__ == "__main__":
    main()
