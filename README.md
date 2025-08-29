# RetinaFace Image Processor and Tagger

This project provides a comprehensive solution for processing and tagging image datasets. It automates HEIC conversion, detects and removes letterboxes, performs face detection for image classification, and generates descriptive tags using a BLIP model.

## Features

-   **HEIC Conversion:** Automatically converts `.HEIC` files to `.png` format.
-   **Letterbox Removal:** Detects and crops black or white letterboxes/pillarboxes from images, overwriting the original file.
-   **Face Detection & Sorting:** Classifies images based on the number of detected faces (0, 1, or multiple) and moves 'Fail' images to a dedicated subdirectory.
-   **BLIP Image Tagging:** Generates descriptive text tags for images in specified subfolders, saving them as `.txt` files.
-   **Configurable:** Easily adjust model parameters and tagging behavior via a YAML configuration file.

## Project Structure

```
retinaface/
├── .gitignore
├── .python-version
├── counter.py          # Handles counting statistics for image processing.
├── face.py             # Contains face detection and letterbox detection/cropping logic.
├── facedetector.py     # (External library component for RetinaFace)
├── main.py             # Orchestrates the entire image processing and tagging workflow.
├── manager.py          # Manages file operations (moving, overwriting, validation).
├── tagger.py           # Encapsulates the BLIP model loading and tag generation logic.
├── heic_convertor.sh   # Shell script for converting HEIC files to PNG.
├── run.sh              # Main entry point script to run the entire process.
├── tagger_config.yaml  # Configuration file for the BLIP tagger.
├── download_model.py   # Helper script to download BLIP models from Hugging Face.
└── README.md           # This documentation file.
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd retinaface
    ```

2.  **Install Dependencies:**
    This project uses `uv` for package management. Ensure you have `uv` installed.
    ```bash
    uv pip install transformers torch Pillow PyYAML
    ```

3.  **Optional: Download BLIP Model Locally**
    For faster loading and offline use, you can download the BLIP model to your local machine. This is highly recommended for repeated use.
    ```bash
    python download_model.py Salesforce/blip-image-captioning-base ./models/blip-base
    ```
    You can choose a different model name or save directory.

## Configuration (`tagger_config.yaml`)

This YAML file allows you to customize the behavior of the BLIP tagger.

```yaml
# --- Model Location ---
# This can be EITHER a model identifier from the Hugging Face Hub 
# (e.g., "Salesforce/blip-image-captioning-base") OR a path to a local 
# directory containing a downloaded model (e.g., "./models/blip-base").
model_path: "Salesforce/blip-image-captioning-base"

# --- Device Settings ---
# 'auto' will use GPU if available, otherwise CPU.
# You can also force it, e.g., 'cuda:0' or 'cpu'.
device: "auto"

# --- Generation Settings ---
# These parameters control the caption generation process.
# See Hugging Face documentation for more details on what these do.
decoding:
  # 'num_beams' > 1 enables beam search. 'num_beams: 1' uses greedy search.
  num_beams: 1
  # 'do_sample: True' enables sampling. Required for 'top_p'.
  do_sample: True
  top_p: 0.7
  max_length: 150
  min_length: 10
  # Note: Current implementation processes one image at a time, 
  # so 'batch_size' here is conceptual for single image generation.
  batch_size: 1

# --- Output Formatting ---
# Add custom text before or after the generated caption.
formatting:
  prefix: ""
  postfix: ""
```

## Usage

Use the `run.sh` script as the main entry point for the entire process.

```bash
./run.sh <target_directory> [--gpu <gpu_id>]
```

-   `<target_directory>`: The absolute path to the directory containing your images and subfolders.
-   `--gpu <gpu_id>` (optional): Specify the ID of the GPU to use (e.g., `0`). If omitted, CPU will be used.

**Example:**

```bash
./run.sh /mnt/DATA/my_image_dataset --gpu 0
```

### Workflow

The `run.sh` script executes `main.py`, which performs the following two phases:

1.  **Image Processing and Sorting Phase:**
    -   Scans the `<target_directory>` and its subfolders for `.HEIC` files and converts them to `.png`.
    -   Detects and removes letterboxes/pillarboxes from images, overwriting the original file with the cropped version.
    -   Performs face detection on images.
    -   Images with 0 or multiple faces, or those with low resolution/errors, are moved to a `Fail` subdirectory within their original location.
    -   Images with exactly one face (and no letterbox) remain in their original location.

2.  **BLIP Tagging Phase:**
    -   After the first phase completes, it iterates through the *immediate subdirectories* of the `<target_directory>` (excluding `Fail` and `letterbox` folders).
    -   For each image found in these subdirectories:
        -   A BLIP model generates a descriptive tag (caption).
        -   The tag is formatted with a dynamic prefix (the name of the image's parent folder) and a configurable postfix.
        -   The final tag content is saved to a `.txt` file with the same base name as the image (e.g., `image.jpg` -> `image.txt`).

## Troubleshooting

-   **`SyntaxError: unmatched ')'`**: Ensure your `main.py` is up-to-date. This was a known issue that has been fixed.
-   **`AttributeError` related to TensorFlow/Keras**: This indicates a version incompatibility between your installed Python packages. Ensure you have installed all dependencies using `uv pip install transformers torch Pillow PyYAML`.
-   **Images not being cropped**: The letterbox detection might be too strict. Try adjusting the `thresh` and `tolerance` parameters in `face.py` (though the current version should be robust).
-   **Model loading issues**: Check your `tagger_config.yaml` for the correct `model_path`. If using a local path, ensure the model was fully downloaded using `download_model.py`.
