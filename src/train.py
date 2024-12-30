import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from utils.config import Config, EarlyStopping
from utils.augmentation import get_augmentation_pipeline
from tqdm import tqdm
from datasets import RetinaDataset
from models.swin_transformer import get_swin_model

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    return correct / total


def main():
    dataset = RetinaDataset(
        csv_path=Config.TRAIN_CSV,
        dir_images=Config.IMAGE_DIR,
        transforms=get_augmentation_pipeline(train=True)
    )

    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataset = Subset(dataset, train_indices.indices)
    val_dataset = Subset(dataset, val_indices.indices)
    val_dataset.dataset.transforms = get_augmentation_pipeline(train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Swin
    model, processor, model_name = get_swin_model(Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1, verbose=True)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    losses = []
    val_accuracies = []
    print(f'DEVICE: {Config.DEVICE}')
    print(f'BATCH_SIZE: {Config.BATCH_SIZE}')
    print(f'LR: {Config.LEARNING_RATE}')
    print(f'Model Name: {model_name}')

    for epoch in tqdm(range(Config.EPOCHS)):
        train_loss = train_epoch(model, train_loader, optimizer, Config.DEVICE)
        val_accuracy = validate(model, val_loader, Config.DEVICE)

        scheduler.step(val_accuracy) 

        losses.append(train_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if early_stopping.step(val_accuracy):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    torch.save(model.state_dict(), f"{Config.MODEL_SAVE_DIR}/{model_name}.pth")

    # Plot Training Loss
    plt.style.use("seaborn-v0_8-paper")  # Scientific style
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label=f'Training Loss - {model_name}')
    plt.title("Training Loss", fontsize=16, weight='bold')
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/saves/training_loss.png", dpi=300)
    plt.show()

    # Plot Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o', label=f'Validation Accuracy - {model_name}')
    plt.title("Validation Accuracy", fontsize=16, weight='bold')
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/saves/validation_accuracy.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()