from transformers import AutoImageProcessor, ConvNextForImageClassification

def get_convnext_model(num_classes):
    model_name = "facebook/convnext-tiny-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ConvNextForImageClassification.from_pretrained(
        model_name,
        num_labels = num_classes,
        ignore_mismatched_sizes=True
    )
    return model, processor, "ConvNeXt"