import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from utils.config import Config
from utils.augmentation import get_augmentation_pipeline
from models.swin_transformer import get_swin_model

# Configuration
path = "/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/models/Swin.pth"
TEST_IMAGES_DIR = "/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/datasets/aptos2019-blindness-detection/test_images"
OUTPUT_CSV = "/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/saves/submission.csv"
DEVICE = Config.DEVICE
validation_pipeline_augment = get_augmentation_pipeline(train = False)

def load_model(weights_path):
    model, _, name = get_swin_model(num_classes=Config.NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model

def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = validation_pipeline_augment(image).unsqueeze(0).to(Config.DEVICE) 
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs.logits, 1)
    return predicted_class.item()

def generate_predictions():
    model = load_model(path)
    results = []

    for filename in tqdm(sorted(os.listdir(TEST_IMAGES_DIR)), desc = "Inference"):
        if filename.endswith(('.png', '.jpg', '.jpeg')): 
            image_path = os.path.join(TEST_IMAGES_DIR, filename)
            prediction = predict(model, image_path)
            results.append({"id_code": os.path.splitext(filename)[0], "diagnosis": prediction})
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print("Predictions finished")

if __name__ == "__main__":
    generate_predictions()