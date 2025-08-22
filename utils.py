import time
import cv2
import numpy as np

class Counter:
    def __init__(self):
        self.stats = {
            "Pass": 0,
            "LowRes": 0,
            "NoFace": 0,
            "MultiFace": 0,
            "NotIMG": 0,
            "Errors": 0
        }

    def increment(self, category):
        if category in self.stats:
            self.stats[category] += 1

    def get_stats(self):
        return self.stats

    def get_total_fails(self):
        return sum(v for k, v in self.stats.items() if k != "Pass")

    def get_summary(self):
        summary = ""
        summary += "==============================" + "\n"
        summary += "작업이 완료되었습니다." + "\n"
        summary += f"총 소요 시간: {self.total_time}" + "\n"
        summary += "==============================" + "\n"
        summary += "분류 결과 통계:" + "\n"
        summary += f"  - Pass: {self.stats['Pass']}개" + "\n"
        summary += f"  - Fail (Low Resolution): {self.stats['LowRes']}개" + "\n"
        summary += f"  - Fail (No Face Detected): {self.stats['NoFace']}개" + "\n"
        summary += f"  - Fail (Multiple Faces Detected): {self.stats['MultiFace']}개" + "\n"
        summary += f"  - Fail (Not an Image File): {self.stats['NotIMG']}개" + "\n"
        if self.stats['Errors'] > 0:
            summary += f"  - Errors (Processing Failed): {self.stats['Errors']}개" + "\n"
        summary += "=============================="
        return summary
        
    def set_total_time(self, total_time):
        self.total_time = time.strftime('%H:%M:%S', time.gmtime(total_time))

def find_crop_coords(img):
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
