import os
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import cv2

SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

class FileManager:
    def __init__(self, target_dir):
        self.target_dir = Path(target_dir)
        self.fail_path = self.target_dir / 'Fail'
        self._create_dirs()

    def _create_dirs(self):
        self.fail_path.mkdir(exist_ok=True)

    def get_image_files(self) -> list[Path]:
        image_files = []
        for root, dirs, files in os.walk(self.target_dir):
            # Exclude 'Fail' directories from further traversal
            dirs[:] = [d for d in dirs if d != 'Fail']
            for filename in files:
                if Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    image_files.append(Path(root) / filename)
        return image_files

    def validate_image(self, image_path: Path):
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width < 1024 or height < 1024:
                    return "LowRes"
        except UnidentifiedImageError:
            return "NotIMG"
        except Exception as e:
            # Log this error properly in the main loop
            return "Errors"
        return "Valid"

    def move_file(self, image_path: Path, result: str):
        if result == "Pass":
            return # Do nothing for passed files

        # For any fail reason, move to the Fail folder
        try:
            fail_dir = image_path.parent / 'Fail'
            fail_dir.mkdir(exist_ok=True)
            shutil.move(str(image_path), str(fail_dir / image_path.name))
        except Exception as e:
            # This exception should be caught and logged in the main loop
            raise e

    def overwrite_image(self, img_data, image_path: Path):
        """Overwrites the original image file with new image data."""
        try:
            # Use the original file extension
            cv2.imwrite(str(image_path), img_data)
        except Exception as e:
            raise e