import torch
from src.utils.clip import cli_model, cli_processor
from PIL import Image


def get_img_embedding(image: Image.Image):
    inputs = cli_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = cli_model.get_image_features(**inputs)

    # If features is a Hugging Face Output object rather than a raw Tensor,
    # extract the embedding attribute dynamically.
    if hasattr(features, "image_embeds"):
        tensor = features.image_embeds
    elif hasattr(features, "pooler_output"):
        tensor = features.pooler_output
    else:
        tensor = features  # It's already a raw tensor

    # Move to CPU, convert to numpy, and flatten it into a clean 1D array
    vector = tensor.cpu().numpy().flatten()
    
    return vector.tolist()