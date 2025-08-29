import yaml
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

class BlipTagger:
    def __init__(self, config_path="tagger_config.yaml"):
        """Initializes the Tagger by loading the model and processor based on the config file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        model_path = self.config.get("model_path", "Salesforce/blip-image-captioning-base")
        device = self.config.get("device", "auto")

        print(f"\nLoading BLIP model from: {model_path}")
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        print(f"BLIP model loaded on device: {self.device}")

    def generate_tag(self, image_path):
        """Generates a descriptive tag (caption) for a single image."""
        try:
            raw_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return None, f"Error opening image {image_path}: {e}"

        # Get decoding parameters from config, with defaults
        decoding_params = self.config.get("decoding", {})
        num_beams = decoding_params.get("num_beams", 1)
        do_sample = decoding_params.get("do_sample", True)
        top_p = decoding_params.get("top_p", 0.7)
        max_length = decoding_params.get("max_length", 150)
        min_length = decoding_params.get("min_length", 10)

        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            num_beams=num_beams,
            do_sample=do_sample,
            top_p=top_p,
            max_length=max_length,
            min_length=min_length
        )
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption, None

    def get_postfix(self):
        """Returns the postfix string from the config."""
        formatting = self.config.get("formatting", {})
        return formatting.get("postfix", "")
