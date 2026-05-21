from transformers import CLIPProcessor, CLIPModel


# =========================
# LOAD CLIP MODEL (ONCE)
# =========================
cli_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
cli_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")