import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.config import Config
from utils.augmentation import get_augmentation_pipeline
from models.swin_transformer import get_swin_model

def load_model(model_path):
    """
    Load the Swin Transformer model with pre-trained weights.

    Args:
        model_path (str): Path to the model's pre-trained weights file.

    Returns:
        model_swin: The loaded Swin Transformer model.
        name: Name of the model architecture.
    """
    model_swin, _, name = get_swin_model(Config.NUM_CLASSES)
    model_swin.load_state_dict(torch.load(model_path, weights_only=True))
    return model_swin, name

def visualize_attention_maps(model_path, image_path, combine_all=False, vizu=False):
    """
    Visualize attention maps for all layers and heads in a Swin Transformer model.

    Args:
        model_path (str): Path to the pre-trained model weights.
        image_path (str): Path to the input image.
        combine_all (bool): If True, combines all attention maps into a single global attention map.
        vizu (bool): If True, prints debugging information about the attention shapes.

    Returns:
        combined_attention_map (np.array): Resized global attention map if combine_all=True.
        all_attention_maps (list): List of attention maps for each layer.
        outputs: Model outputs, including logits and attention data.
    """
    # Load the model
    model, name_model = load_model(model_path)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    valid_pipeline_augmentation = get_augmentation_pipeline(train=False)
    image_tensor = valid_pipeline_augmentation(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor, output_attentions=True)
        attentions = outputs.attentions
        if vizu:
            print(f"Number of attention layers: {len(attentions)}")
            print(f"Attention shapes: {[a.shape for a in attentions]}")

    all_attention_maps = []

    for n, attention_layer in enumerate(attentions):
        attention_layer = attention_layer[0] 
        num_heads = attention_layer.size(0)  
        layer_attention_maps = []

        for i in range(num_heads):
            attention_map = attention_layer[i].mean(dim=0) 
            layer_attention_maps.append(attention_map)

        # Combine all head attentions into a single map for the layer
        combined_layer_attention = torch.stack(layer_attention_maps).mean(dim=0)
        all_attention_maps.append(combined_layer_attention)

    if combine_all:
        global_attention_map = torch.stack(all_attention_maps).mean(dim=0)
        grid_size = int(np.sqrt(global_attention_map.size(0)))
        global_attention_map = global_attention_map.reshape(grid_size, grid_size).numpy()
        global_attention_map = (global_attention_map - global_attention_map.min()) / (
                    global_attention_map.max() - global_attention_map.min())

        global_attention_map_resized = np.array(
            Image.fromarray(global_attention_map).resize(image.size, resample=Image.BILINEAR)
        )
        return global_attention_map_resized, all_attention_maps, outputs

    return all_attention_maps, outputs

def plot_global_attention(image, global_attention_map, save_path = "/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/saves/global_attention.png"):
    """
    Plot the original image, global attention map, and their overlay.

    Args:
        image (PIL.Image): Input image.
        global_attention_map (np.array): Combined global attention map.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(global_attention_map, cmap='jet')
    ax[1].set_title("Global Attention Map")
    ax[1].axis('off')

    ax[2].imshow(image)
    ax[2].imshow(global_attention_map, cmap='jet', alpha=0.3)
    ax[2].set_title("Overlay: Image + Attention")
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_layerwise_attention(image, all_attention_maps):
    """
    Plot the attention maps for all layers.

    Args:
        image (PIL.Image): Input image.
        all_attention_maps (list): List of attention maps for each layer.
    """
    num_layers = len(all_attention_maps)
    _, axes = plt.subplots(num_layers, 3, figsize=(15, 5 * num_layers))

    for idx, attention_map in enumerate(all_attention_maps):
        grid_size = int(np.sqrt(attention_map.size(0)))
        attention_map = attention_map.reshape(grid_size, grid_size).numpy()
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        attention_map_resized = np.array(
            Image.fromarray(attention_map).resize(image.size, resample=Image.BILINEAR)
        )

        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f"Layer {idx + 1}: Original Image")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(attention_map_resized, cmap='jet')
        axes[idx, 1].set_title(f"Layer {idx + 1}: Attention Map")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(image)
        axes[idx, 2].imshow(attention_map_resized, cmap='jet', alpha=0.3)
        axes[idx, 2].set_title(f"Layer {idx + 1}: Overlay")
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig("/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/saves/layers_attention.png")
    plt.show()

if __name__ == '__main__':
    # Paths
    model_path = "/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/models/Swin.pth"
    image_path = "/Users/armandbryan/Documents/challenges/Computer Vision Projects Expo 2024/datasets/aptos2019-blindness-detection/test_images/0a2b5e1a0be8.png"

    image = Image.open(image_path).convert('RGB')
    global_attention, all_attention_maps, outputs = visualize_attention_maps(
        model_path=model_path,
        image_path=image_path,
        combine_all=True,
        vizu=True
    )

    plot_global_attention(image, global_attention)
    plot_layerwise_attention(image, all_attention_maps)

    # Display predictions
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    predicted_prob = torch.max(probs, dim=-1).values.item()
    print(f"Predicted Class: {predicted_class}, Probability: {predicted_prob:.2f}")