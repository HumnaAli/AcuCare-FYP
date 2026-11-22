# seg.py
# FCM-based segmentation pipeline with preprocessing (denoising + CLAHE)

import os
import cv2
import glob
import numpy as np
import shutil
import random
from skimage.morphology import remove_small_objects, disk
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
import skfuzzy as fuzz

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def segment_with_fcm(img_bgr, L_thresh=220, B_thresh=150):
    # Preprocessing: Denoising and CLAHE on L-channel
    img_denoised = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    img_lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    img_lab = cv2.merge([L, a, b])

    # FCM clustering on a* and b*
    H, W = a.shape
    ab = np.vstack((a.flatten(), b.flatten()))
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(ab, 3, 2.0, error=0.005, maxiter=1000)
    cluster_labels = np.argmax(u, axis=0).reshape(H, W)
    lesion_class = int(np.argmax([a[cluster_labels == i].mean() for i in range(3)]))
    mask = (cluster_labels == lesion_class).astype(np.uint8)

    # Morphological cleanup and pus enhancement
    opened = binary_opening(mask, structure=disk(3)).astype(np.uint8)
    closed = binary_closing(opened, structure=disk(5)).astype(np.uint8)
    pus = ((L > L_thresh) & (b > B_thresh)).astype(np.uint8)
    combined = np.maximum(closed, pus)
    final = binary_opening(combined, structure=disk(1)).astype(bool)
    final = binary_closing(final, structure=disk(1)).astype(bool)
    final = remove_small_objects(final, min_size=200)
    final_filled = binary_fill_holes(final)
    final_clean = remove_small_objects(final_filled, min_size=200).astype(np.uint8) * 255
    roi = img_bgr.copy()
    roi[final_clean == 0] = 0

    return final_clean

def run_segmentation(raw_folder='compiled', split_folder='split_dataset', out_root='seg_fcm'):
    roi_dir = os.path.join(out_root, 'grayscale_roi')
    mask_dir = os.path.join(out_root, 'lesion_masks')
    overlay_dir = os.path.join(out_root, 'overlay')
    for d in [roi_dir, mask_dir, overlay_dir]:
        ensure_dir(d)


    for split in ("train", "test"):
        for cls in ("nodules", "papule", "pustules"):
            split_path = os.path.join(split_folder, split, cls)
            for img_fp in glob.glob(os.path.join(split_path, '*.*')):
                fname = f"{split}_" + os.path.basename(img_fp)
                img = cv2.imread(img_fp)
                if img is None:
                    continue
                img = cv2.resize(img, (150, 150))
                mask = segment_with_fcm(img)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                masked = img.copy()
                masked[mask == 0] = 0
                cv2.imwrite(os.path.join(roi_dir, fname), masked)
                cv2.imwrite(os.path.join(mask_dir, fname), mask)
                overlay = img.copy()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=2)
                cv2.imwrite(os.path.join(overlay_dir, fname), overlay)
                print(f"Segmented: {fname}")

if __name__ == '__main__':
    run_segmentation()