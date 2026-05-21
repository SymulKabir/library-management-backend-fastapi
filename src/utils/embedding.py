
import torch
from src.utils.clip import cli_model, cli_processor
from PIL import Image


def get_img_embedding(image: Image.Image):
    inputs = cli_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = cli_model.get_image_features(**inputs)

    vector = features[0].cpu().numpy()
    return vector.tolist()

