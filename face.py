from retinaface import RetinaFace
import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self._model = RetinaFace.build_model()

    def _check_letterbox(self, img):
        """Checks for black or white letterboxing or pillarboxing in an image."""
        height, width, _ = img.shape
        BLACK_THRESH = 10
        WHITE_THRESH = 245
        edge_fraction = 0.1

        # Letterboxing check
        top_rows = int(height * edge_fraction)
        bottom_rows_start = height - top_rows
        top_band = img[0:top_rows, :]
        bottom_band = img[bottom_rows_start:height, :]

        if np.mean(top_band) < BLACK_THRESH and np.mean(bottom_band) < BLACK_THRESH and np.std(top_band) < BLACK_THRESH and np.std(bottom_band) < BLACK_THRESH:
            return True
        if np.mean(top_band) > WHITE_THRESH and np.mean(bottom_band) > WHITE_THRESH and np.std(top_band) < BLACK_THRESH and np.std(bottom_band) < BLACK_THRESH:
            return True

        # Pillarboxing check
        side_cols = int(width * edge_fraction)
        right_cols_start = width - side_cols
        left_band = img[:, 0:side_cols]
        right_band = img[:, right_cols_start:width]

        if np.mean(left_band) < BLACK_THRESH and np.mean(right_band) < BLACK_THRESH and np.std(left_band) < BLACK_THRESH and np.std(right_band) < BLACK_THRESH:
            return True
        if np.mean(left_band) > WHITE_THRESH and np.mean(right_band) > WHITE_THRESH and np.std(left_band) < BLACK_THRESH and np.std(right_band) < BLACK_THRESH:
            return True

        return False

    def process_image(self, img_path):
        """Reads an image, detects faces, and checks for letterboxing."""
        img = cv2.imread(img_path)
        if img is None:
            return None, None, True # Treat as an error case

        faces = RetinaFace.detect_faces(img_path, model=self._model)
        has_letterbox = self._check_letterbox(img)

        return faces, img, has_letterbox