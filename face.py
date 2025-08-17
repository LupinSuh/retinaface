from retinaface import RetinaFace
import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self):
        self._model = RetinaFace.build_model()

    def _find_crop_coords(self, img):
        """Finds the coordinates of the non-letterboxed content using a robust method."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Determine border color by checking corners
        corner_pixels = [gray[0, 0], gray[0, w-1], gray[h-1, 0], gray[h-1, w-1]]
        mean_corner_color = np.mean(corner_pixels)

        if mean_corner_color < 128: # Assume black letterbox
            thresh = 20 # More lenient black threshold
            is_dark_border = True
        else: # Assume white letterbox
            thresh = 235 # More lenient white threshold
            is_dark_border = False
        
        tolerance = 0.02 # Allow up to 2% of pixels to be different

        # --- Scan based on detected border color ---
        y_start = 0
        for i in range(h):
            row = gray[i, :]
            if is_dark_border:
                if (np.count_nonzero(row > thresh) / w) > tolerance: y_start = i; break
            else:
                if (np.count_nonzero(row < thresh) / w) > tolerance: y_start = i; break
        
        y_end = h
        for i in range(h - 1, -1, -1):
            row = gray[i, :]
            if is_dark_border:
                if (np.count_nonzero(row > thresh) / w) > tolerance: y_end = i + 1; break
            else:
                if (np.count_nonzero(row < thresh) / w) > tolerance: y_end = i + 1; break

        x_start = 0
        for i in range(w):
            col = gray[:, i]
            if is_dark_border:
                if (np.count_nonzero(col > thresh) / h) > tolerance: x_start = i; break
            else:
                if (np.count_nonzero(col < thresh) / h) > tolerance: x_start = i; break

        x_end = w
        for i in range(w - 1, -1, -1):
            col = gray[:, i]
            if is_dark_border:
                if (np.count_nonzero(col > thresh) / h) > tolerance: x_end = i + 1; break
            else:
                if (np.count_nonzero(col < thresh) / h) > tolerance: x_end = i + 1; break

        # Check if the crop is significant and valid
        if x_start < x_end and y_start < y_end and (x_start > 0 or y_start > 0 or x_end < w or y_end < h):
            return y_start, y_end, x_start, x_end
        else:
            return None

    def process_image(self, img_path):
        """Reads an image, crops letterbox, and detects faces."""
        img = cv2.imread(img_path)
        if img is None:
            return None, None, False

        was_cropped = False
        coords = self._find_crop_coords(img)
        
        if coords is not None:
            h, w, _ = img.shape
            y_start, y_end, x_start, x_end = coords
            
            if y_end > y_start and x_end > x_start:
                print(f"  - Cropping: {os.path.basename(img_path)}. Original: {w}x{h}, New: {(x_end-x_start)}x{(y_end-y_start)}")
                img = np.ascontiguousarray(img[y_start:y_end, x_start:x_end])
                was_cropped = True

        faces = RetinaFace.detect_faces(img, model=self._model)

        return faces, img, was_cropped