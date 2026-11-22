import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops
from skimage import measure

# === Settings ===
input_root = r'C:\Users\Dell\Desktop\defence\seg_results_test_gray\grayscale_roi'
GLCM_LEVELS = 8
LBP_P = 8
LBP_R = 1
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0]


def extract_features(img_gray, mask, img_rgb):
    import numpy as np
    import cv2
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage import measure

    GLCM_LEVELS = 8
    GLCM_DISTANCES = [1]
    GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    LBP_P = 8
    LBP_R = 1

    # 1) GLCM features (averaged across 4 angles)
    quantized = (img_gray // (256 // GLCM_LEVELS)).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    contrast    = np.mean(graycoprops(glcm, prop="contrast"))
    correlation = np.mean(graycoprops(glcm, prop="correlation"))
    energy      = np.mean(graycoprops(glcm, prop="energy"))
    homogeneity = np.mean(graycoprops(glcm, prop="homogeneity"))

    # 2) Grayscale stats
    masked_pixels = img_gray[mask == 255]
    mean_L = float(masked_pixels.mean()) if masked_pixels.size > 0 else 0.0
    std_L  = float(masked_pixels.std())  if masked_pixels.size > 0 else 0.0

    # 3) Basic shape features
    props = measure.regionprops((mask == 255).astype(np.uint8))
    if props:
        area = float(props[0].area)
        perimeter = float(props[0].perimeter) if props[0].perimeter > 0 else 1.0
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
    else:
        area = perimeter = circularity = 0.0

    # 4) LBP variance
    lbp = local_binary_pattern(img_gray, P=LBP_P, R=LBP_R, method="uniform")
    lbp_vals = lbp[mask == 255]
    lbp_variance = float(lbp_vals.var()) if lbp_vals.size > 0 else 0.0

    # 5) HSV color features
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    masked_h = img_hsv[:, :, 0][mask == 255]
    masked_s = img_hsv[:, :, 1][mask == 255]
    masked_v = img_hsv[:, :, 2][mask == 255]
    mean_h = masked_h.mean() if masked_h.size > 0 else 0.0
    mean_s = masked_s.mean() if masked_s.size > 0 else 0.0
    mean_v = masked_v.mean() if masked_v.size > 0 else 0.0
    std_h  = masked_h.std() if masked_h.size > 0 else 0.0
    std_s  = masked_s.std() if masked_s.size > 0 else 0.0
    std_v  = masked_v.std() if masked_v.size > 0 else 0.0

    # Final feature vector
    feats = np.array([
        contrast, correlation, energy, homogeneity,
        mean_L, std_L,
        area, perimeter, circularity,
        lbp_variance,
        mean_h, mean_s, mean_v,
        std_h, std_s, std_v
    ], dtype=np.float32)

    return feats


def parse_split_class(fname):
    fname = fname.lower()
    # Get split
    if fname.startswith('train_'):
        split = 'train'
    elif fname.startswith('test_'):
        split = 'test'

    else:
        raise ValueError(f"Unknown split in filename: {fname}")

    # Remove split part
    after_split = fname.split('_', 1)[1]

    # Assign class label based on pattern after split
    if after_split.startswith('n'):        
        label = 0
    elif after_split.startswith('pa'):     
        label = 1
    elif 'p' in after_split or 'acne-pustular' in after_split or after_split.startswith('p'):
        label = 2
    else:
        raise ValueError(f"Unknown class in filename: {fname}")

    return split, label




# ==== Main Extraction Loop ====
X_train, y_train, train_filenames = [], [], []
X_test, y_test, test_filenames = [], [], []


for fname in os.listdir(input_root):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        continue
    try:
        split, label = parse_split_class(fname)
    except Exception as e:
        print(f"[ERROR] {e}")
        continue

    img_path = os.path.join(input_root, fname)
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_rgb  = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_gray is None or img_rgb is None:
        print(f"[ERROR] Could not read {img_path}")
        continue
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    mask[img_gray > 0] = 255
    feats = extract_features(img_gray, mask, img_rgb)

    if split == 'train':
        X_train.append(feats)
        y_train.append(label)
        train_filenames.append(fname)
    elif split == 'test':
        X_test.append(feats)
        y_test.append(label)
        test_filenames.append(fname)


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

train_filenames = np.array(train_filenames)
test_filenames = np.array(test_filenames)


print(f"Train: {X_train.shape} | Test: {X_test.shape}")

np.savez('features_kmeans.npz',
         X_train=X_train, y_train=y_train,
         X_test=X_test, y_test=y_test,

         train_filenames=train_filenames,
         test_filenames=test_filenames)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
