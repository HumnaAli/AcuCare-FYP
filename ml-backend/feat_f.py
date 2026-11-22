import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops
from skimage import measure

# === Settings ===
input_root = r'C:\Users\Dell\Desktop\defence\seg_fcm\grayscale_roi'
GLCM_LEVELS = 8
LBP_P = 8
LBP_R = 1
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0]

def gabor_feats(img_gray, mask):
    feats = []
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    sigmas = [1, 2]
    for theta in thetas:
        for sigma in sigmas:
            kernel = cv2.getGaborKernel((9, 9), sigma, theta, 8, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(img_gray, cv2.CV_8UC3, kernel)
            region = filtered[mask == 255]
            feats.append(region.mean() if region.size > 0 else 0.0)
            feats.append(region.std() if region.size > 0 else 0.0)
    return feats  # 4 orientations × 2 sigmas × 2 (mean+std) = 16 features

def extract_features(img_gray, mask, img_rgb):
    # 1) Extended GLCM features (4 angles)
    quantized = (img_gray // (256 // GLCM_LEVELS)).astype(np.uint8)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(quantized, distances=GLCM_DISTANCES, angles=angles, levels=GLCM_LEVELS, symmetric=True, normed=True)
    contrast    = np.mean(graycoprops(glcm, prop="contrast"))
    correlation = np.mean(graycoprops(glcm, prop="correlation"))
    energy      = np.mean(graycoprops(glcm, prop="energy"))
    homogeneity = np.mean(graycoprops(glcm, prop="homogeneity"))

    # 2) Grayscale stats inside mask
    masked_pixels = img_gray[mask == 255]
    mean_L = float(masked_pixels.mean()) if masked_pixels.size > 0 else 0
    std_L = float(masked_pixels.std()) if masked_pixels.size > 0 else 0

    # 3) Shape stats (+ more features)
    props = measure.regionprops((mask == 255).astype(np.uint8))
    if props:
        area = float(props[0].area)
        perimeter = float(props[0].perimeter) if props[0].perimeter > 0 else 1.0
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
        solidity = props[0].solidity if hasattr(props[0], 'solidity') else 0.0
        eccentricity = props[0].eccentricity if hasattr(props[0], 'eccentricity') else 0.0
        extent = props[0].extent if hasattr(props[0], 'extent') else 0.0
        major_axis = props[0].major_axis_length if hasattr(props[0], 'major_axis_length') else 0.0
        minor_axis = props[0].minor_axis_length if hasattr(props[0], 'minor_axis_length') else 0.0
        bbox = props[0].bbox
        aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) if (bbox[3] - bbox[1]) != 0 else 0.0
    else:
        area = perimeter = circularity = 0.0
        solidity = eccentricity = extent = 0.0
        major_axis = minor_axis = aspect_ratio = 0.0

    # 4) Edge density (Canny)
    edges = cv2.Canny(img_gray, 50, 150)
    edge_pixels = np.count_nonzero(edges[mask == 255])
    edge_density = float(edge_pixels) / (area + 1e-6) if area > 0 else 0

    # 5) LBP variance
    lbp = local_binary_pattern(img_gray, P=LBP_P, R=LBP_R, method="uniform")
    lbp_vals = lbp[mask == 255]
    lbp_variance = float(lbp_vals.var()) if lbp_vals.size > 0 else 0

    # 9) Gabor features
    gabor = gabor_feats(img_gray, mask)  # 16 features

    feats = np.array([
        contrast, correlation, energy, homogeneity,
        mean_L, std_L,
        area, perimeter, circularity,
        edge_density, lbp_variance,
        solidity, eccentricity, extent, major_axis, minor_axis, aspect_ratio,  # shape
        *gabor
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

np.savez('feature_fmeans.npz',
         X_train=X_train, y_train=y_train,
         X_test=X_test, y_test=y_test,

         train_filenames=train_filenames,
         test_filenames=test_filenames)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
