import argparse
import os
from transformers import BlipProcessor, BlipForConditionalGeneration

def download_model(model_name, save_directory):
    """Downloads a BLIP model and its processor to a specified directory."""
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"Created directory: {save_directory}")

    print(f"Downloading model and processor for '{model_name}'...")

    try:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)

        processor.save_pretrained(save_directory)
        model.save_pretrained(save_directory)

        print(f"Model successfully downloaded and saved to {save_directory}")
        print(f"You can now update 'model_path' in your tagger_config.yaml to:")
        print(f"{os.path.abspath(save_directory)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Hugging Face BLIP model.")
    parser.add_argument(
        "model_name", 
        type=str, 
        help="The model identifier from the Hugging Face Hub (e.g., 'Salesforce/blip-image-captioning-base')."
    )
    parser.add_argument(
        "save_directory", 
        type=str, 
        help="The local path to save the model files to (e.g., './models/blip-base')."
    )
    args = parser.parse_args()

    download_model(args.model_name, args.save_directory)
