import os
import cv2
import glob
import numpy as np
import shutil
import random
from skimage.morphology import remove_small_objects, disk
from scipy.ndimage import binary_opening, binary_closing
from sklearn.cluster import KMeans

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def split_dataset(raw_root, split_root, split_ratio=(0.70, 0.30), seed=42):
    classes = ["nodules", "papule", "pustules"]
    rng = random.Random(seed)
    for cls in classes:
        src_dir = os.path.join(raw_root, cls)
        files = sorted([f for ext in ('*.jpg', '*.jpeg', '*.png', '*tif', '*tiff') for f in glob.glob(os.path.join(src_dir, ext))])
        rng.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * split_ratio[0])
        splits = [("train", files[:n_train]), ("test", files[n_train:])]
        for split_name, split_files in splits:
            dst_dir = os.path.join(split_root, split_name, cls)
            ensure_dir(dst_dir)
            for f in split_files:
                shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))

import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.morphology import disk, binary_opening, binary_closing, remove_small_objects
from scipy.ndimage import binary_fill_holes

def segment_acne_lesion(img_bgr, L_thresh=220, B_thresh=150):
    img_resized = cv2.resize(img_bgr, (150, 150))
    img_denoised = cv2.bilateralFilter(img_resized, 9, 75, 75)
    img_lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(img_lab)

    # CLAHE on L*
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    enhanced_lab = cv2.merge([L_clahe, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)  # For matplotlib

    # KMeans segmentation
    H, W = L.shape
    ab = np.stack([a.flatten(), b.flatten()], axis=1).astype(np.float32)
    km = KMeans(n_clusters=3, random_state=0).fit(ab)
    labels = km.labels_.reshape(H, W)
    means = [float(a[labels == c].mean()) for c in range(3)]
    raw = (labels == int(np.argmax(means))).astype(np.uint8)

    opened = binary_opening(raw, footprint=disk(3)).astype(np.uint8)
    closed = binary_closing(opened, footprint=disk(5)).astype(np.uint8)
    pus = ((L_clahe > L_thresh) & (b > B_thresh)).astype(np.uint8)
    combined = np.maximum(closed, pus)
    final = binary_opening(combined, footprint=disk(1)).astype(bool)
    final = binary_closing(final, footprint=disk(1)).astype(bool)

    # Fill holes inside the lesion mask
    final_filled = binary_fill_holes(final)
    final_clean = remove_small_objects(final_filled, min_size=200).astype(np.uint8) * 255
    roi = img_resized.copy()
    roi[final_clean == 0] = 0

    return final_clean



def run_segmentation(raw_folder='compiled', split_folder='split_dataset', out_root='seg_results_test_gray'):
    roi_dir = os.path.join(out_root, 'grayscale_roi')
    mask_dir = os.path.join(out_root, 'lesion_masks')
    overlay_dir = os.path.join(out_root, 'overlay')
    for d in [roi_dir, mask_dir, overlay_dir]:
        ensure_dir(d)

    # Step 1: Split dataset
    required_splits = [
        os.path.join(split_folder, s, c)
        for s in ("train", "test")
        for c in ("nodules", "papules", "pustules")
    ]
    if not all(os.path.isdir(p) and os.listdir(p) for p in required_splits):
        print("Splitting dataset...")
        split_dataset(raw_folder, split_folder)
    else:
        print("Split already exists. Skipping split.")

    # Step 2: Segmentation
    for split in ("train", "test"):
        for cls in ("nodules", "papule", "pustules"):
            split_path = os.path.join(split_folder, split, cls)
            for img_fp in glob.glob(os.path.join(split_path, '*.*')):
                fname = f"{split}_" + os.path.basename(img_fp)
                img = cv2.imread(img_fp)
                if img is None:
                    continue
                img = cv2.resize(img, (150, 150))
                mask = segment_acne_lesion(img)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                masked = img.copy()
                masked[mask == 0] = 0
                # Save grayscale ROI
                cv2.imwrite(os.path.join(roi_dir, fname), masked)
                # Save lesion mask
                cv2.imwrite(os.path.join(mask_dir, fname), mask)
                # Save overlay
                overlay = img.copy()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=2)
                cv2.imwrite(os.path.join(overlay_dir, fname), overlay)
                print(f"Segmented: {fname}")

if __name__ == '__main__':
    run_segmentation()
