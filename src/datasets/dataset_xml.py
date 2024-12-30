import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


class RetinopathyDiabetDatasetMultiLabel:
    """
    Dataset pour la segmentation multi-classes avec labels.
    """
    def __init__(self, txt_paths, BASE_PATH, label_mapping):
        self.data = np.loadtxt(txt_paths, dtype=str)
        self.BASE_PATH = BASE_PATH
        self.label_mapping = label_mapping
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entries = self.data[idx]
        img_path = os.path.join(self.BASE_PATH, entries[0])
        ann_paths = [os.path.join(self.BASE_PATH, ann) for ann in entries[1:]]
        
        # Charger l'image
        image = np.array(Image.open(img_path).convert('RGB'))

        # Créer un mask vide (même taille que l'image, initialisé à 0)
        mask = Image.new('I', (image.shape[1], image.shape[0]), 0)  # Mode 'I' pour valeurs entières
        draw = ImageDraw.Draw(mask)

        # Charger les annotations et dessiner sur le mask
        for xml_path in ann_paths:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for marking in root.find('markinglist').findall('marking'):
                # Obtenir le label
                label = marking.find('markingtype')
                if label is None or label.text not in self.label_mapping:
                    continue
                class_id = self.label_mapping[label.text]  # Convertir le label en ID de classe

                # Dessiner les cercles
                circleregion = marking.find('circleregion')
                if circleregion is not None:
                    centroid = circleregion.find('centroid').find('coords2d').text
                    x, y = map(int, centroid.split(','))
                    radius = int(circleregion.find('radius').text)
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=class_id)

                # Dessiner les polygones
                polygonregion = marking.find('polygonregion')
                if polygonregion is not None:
                    points = []
                    for coord in polygonregion.findall('coords2d'):
                        x, y = map(int, coord.text.split(','))
                        points.append((x, y))
                    draw.polygon(points, fill=class_id)
        
        # Retourner l'image et le mask
        mask = np.array(mask)  # Convertir en tableau numpy
        return image, mask


# Tester le dataset
if __name__ == '__main__':
    label_mapping = {
        "Red_small_dots": 1,
        "Hard_exudates": 2,
        "Haemorrhages": 3
        # Ajoutez d'autres labels si nécessaire
    }
    dataset = RetinopathyDiabetDatasetMultiLabel(
        "/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/datasets/DiaRetDB1 V2.1/ddb1_v02_01_train.txt",
        "/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/datasets/DiaRetDB1 V2.1",
        label_mapping
    )
    image, mask = dataset[0]
    
    # Afficher l'image et le mask
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='jet')  # Utilisez une carte de couleurs pour visualiser les classes
    plt.show()