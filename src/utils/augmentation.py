import torchvision.transforms as transforms

def get_augmentation_pipeline(train = True):
    """
    Returns a data augmentation pipeline for training and validation images.
    """
    if train:
        # Training Augmentations
        train_transforms = transforms.Compose([
            transforms.Resize((300, 300)),         
            transforms.RandomCrop((224, 224)),            
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=(3, 3)), 
            transforms.ToTensor(),                
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   
        ])

        return train_transforms

    # Validation Augmentations
    val_transforms = transforms.Compose([
        transforms.Resize((300, 300)),        
        transforms.CenterCrop((224, 224)),     
        transforms.ToTensor(),            
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    
    return val_transforms