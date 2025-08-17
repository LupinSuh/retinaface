import cv2
import numpy as np
import argparse

def check_letterbox(image_path):
    """Checks for black or white letterboxing or pillarboxing in an image."""
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not read image file."

    height, width, _ = img.shape

    # Define thresholds for black and white
    # (allowing for some compression artifacts)
    BLACK_THRESH = 10
    WHITE_THRESH = 245

    # Define how much of the edge to check (e.g., 10%)
    edge_fraction = 0.1

    # --- Check for horizontal bars (Letterboxing) ---
    top_rows = int(height * edge_fraction)
    bottom_rows_start = height - top_rows

    top_band = img[0:top_rows, :]
    bottom_band = img[bottom_rows_start:height, :]

    # Check for black letterbox
    if np.mean(top_band) < BLACK_THRESH and np.mean(bottom_band) < BLACK_THRESH:
        # To be more certain, check if the bands are almost a solid color
        if np.std(top_band) < BLACK_THRESH and np.std(bottom_band) < BLACK_THRESH:
            return "Black letterbox (top/bottom) detected."

    # Check for white letterbox
    if np.mean(top_band) > WHITE_THRESH and np.mean(bottom_band) > WHITE_THRESH:
        if np.std(top_band) < BLACK_THRESH and np.std(bottom_band) < BLACK_THRESH: # std dev should still be low
            return "White letterbox (top/bottom) detected."

    # --- Check for vertical bars (Pillarboxing) ---
    side_cols = int(width * edge_fraction)
    right_cols_start = width - side_cols

    left_band = img[:, 0:side_cols]
    right_band = img[:, right_cols_start:width]

    # Check for black pillarbox
    if np.mean(left_band) < BLACK_THRESH and np.mean(right_band) < BLACK_THRESH:
        if np.std(left_band) < BLACK_THRESH and np.std(right_band) < BLACK_THRESH:
            return "Black pillarbox (left/right) detected."

    # Check for white pillarbox
    if np.mean(left_band) > WHITE_THRESH and np.mean(right_band) > WHITE_THRESH:
        if np.std(left_band) < BLACK_THRESH and np.std(right_band) < BLACK_THRESH:
            return "White pillarbox (left/right) detected."

    return "No letterbox detected."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for letterboxing or pillarboxing in an image.")
    parser.add_argument("image_path", type=str, help="The path to the image file.")
    args = parser.parse_args()

    result = check_letterbox(args.image_path)
    print(result)
