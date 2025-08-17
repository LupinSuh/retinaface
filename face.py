from retinaface import RetinaFace
import cv2


class FaceDetector:
    def __init__(self):
        self._model = RetinaFace.build_model()

    def detect_faces(self, img_path):
        img = cv2.imread(img_path)
        faces = RetinaFace.detect_faces(img_path, model=self._model)
        return faces, img

    def draw_faces(self, img, faces):
        for face_key, face_info in faces.items():
            x1, y1, x2, y2 = face_info['facial_area']
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        return img
