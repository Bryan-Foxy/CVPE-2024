import os
import torch
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class RetinaDataset(Dataset):
    """
    Prepare the dataset for the classification model.
    """
    def __init__(self, csv_path, dir_images, transforms=None):
        self.csv = pd.read_csv(csv_path)
        image_paths = glob.glob(f"{dir_images}/*.png")
        self.data = {'images': [], 'labels': []}

        for img_path in image_paths:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            if filename in self.csv['id_code'].values:
                index = self.csv[self.csv['id_code'] == filename].index[0]
                self.data['images'].append(img_path)
                self.data['labels'].append(int(self.csv['diagnosis'][index]))

        self.transforms = transforms

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, index):
        img_path = self.data['images'][index]
        img = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.data['labels'][index], dtype=torch.long)
        
        if self.transforms:
            img = self.transforms(img)
        return img, label
