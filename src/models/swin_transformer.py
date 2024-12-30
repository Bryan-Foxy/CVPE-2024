from transformers import AutoImageProcessor, SwinForImageClassification

def get_swin_model(num_classes):
    model_name = "microsoft/swin-tiny-patch4-window7-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SwinForImageClassification.from_pretrained(
        model_name,
        num_labels = num_classes,
        ignore_mismatched_sizes=True
    )
    return model, processor, "Swin"