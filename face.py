from retinaface.pre_trained_models import get_model
import cv2
import numpy as np
import os
import utils

class FaceDetector:
    def __init__(self, device):
        self._model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
        self._model.eval()

    def process_image(self, img_path):
        """Reads an image, crops letterbox, and detects faces.""" 
        img = cv2.imread(img_path)
        if img is None:
            return None, None, False

        was_cropped = False
        coords = utils.find_crop_coords(img)
        
        if coords is not None:
            h, w, _ = img.shape
            y_start, y_end, x_start, x_end = coords
            
            if y_end > y_start and x_end > x_start:
                # print(f"  - Cropping: {os.path.basename(img_path)}. Original: {w}x{h}, New: {(x_end-x_start)}x{(y_end-y_start)}")
                img = np.ascontiguousarray(img[y_start:y_end, x_start:x_end])
                was_cropped = True

        faces = self._model.predict_jsons(img)

        return faces, img, was_cropped
