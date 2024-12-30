import torch

class Config:
    NUM_CLASSES = 5
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")       
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    EPOCHS = 15
    TRAIN_CSV = "/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/datasets/aptos2019-blindness-detection/train.csv"
    IMAGE_DIR = "/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/datasets/aptos2019-blindness-detection/train_images"
    MODEL_SAVE_DIR = "/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/models"

class ClassMapper:
    def __init__(self):
        self.class2name = {
            0: "No DR",
            1: "Mild",
            2: "Moderate",
            3: "Severe",
            4: "Proliferative DR"
        }
        self.name2class = {
            name: class_id for class_id, name in self.class2name.items()
        }
    
    def get_class_name(self, class_id):
        return self.class2name.get(class_id, "Unknow class id")
    def get_class_id(self, class_name):
        return self.name2class.get(class_name, "Unknow class name")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_accuracy = -float("inf")

    def step(self, val_accuracy):
        if val_accuracy > self.best_val_accuracy + self.min_delta:
            self.best_val_accuracy = val_accuracy
            self.counter = 0
        else:
            self.counter += 1
        return self.counter > self.patience