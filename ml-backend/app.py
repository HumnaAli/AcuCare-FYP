import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
import joblib
from skimage.morphology import remove_small_objects, disk
from scipy.ndimage import binary_opening, binary_closing
from sklearn.preprocessing import LabelBinarizer
import skfuzzy as fuzz
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import measure
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

GLCM_LEVELS = 8
LBP_P = 8
LBP_R = 1
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0]


# ============== ELM Function ==============

class ELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, hidden_neurons=50, activation='sigmoid', random_state=0, C=1e3):
        self.input_dim = input_dim
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.random_state = random_state
        self.C = C

    def _activate(self, X):
        H_raw = X.dot(self.w_hidden) + self.b_hidden
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-H_raw))
        elif self.activation == "relu":
            return np.maximum(0, H_raw)
        elif self.activation == "rbf":
            H = np.zeros((X.shape[0], self.hidden_neurons))
            for i in range(self.hidden_neurons):
                center = self.w_hidden[:, i]
                sigma = 1.0
                dist = np.linalg.norm(X - center, axis=1)
                H[:, i] = np.exp(-(dist ** 2) / (2 * sigma ** 2))
            return H
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        lb = LabelBinarizer()
        Y_onehot = lb.fit_transform(y)
        self.label_binarizer_ = lb

        rng = np.random.RandomState(self.random_state)
        self.input_dim_ = X.shape[1] if self.input_dim is None else self.input_dim
        self.w_hidden = rng.randn(self.input_dim_, self.hidden_neurons).astype(np.float32)
        self.b_hidden = rng.randn(self.hidden_neurons).astype(np.float32)

        H = self._activate(X)
        lambda_identity = np.eye(H.shape[1]) / self.C
        self.beta_output = np.linalg.solve(H.T @ H + lambda_identity, H.T @ Y_onehot)
        return self

    def predict_proba(self, X):
        H = self._activate(X)
        scores = H.dot(self.beta_output)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    

# ============== Load ELM pipeline ==============
PIPELINE_PATH = "elm_fmeans.joblib"
if not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError(f"Cannot find '{PIPELINE_PATH}' in the working directory.")

pipeline = joblib.load(PIPELINE_PATH)
elm = pipeline['model']
scaler = pipeline['scaler']
selector = pipeline['selector']

class_names = ["nodules", "papule", "pustules"]

# ============== FCM-based segmentation ==============
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.morphology import disk, binary_opening, binary_closing, remove_small_objects
from scipy.ndimage import binary_fill_holes

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
    opened = binary_opening(mask, footprint=disk(3)).astype(np.uint8)
    closed = binary_closing(opened, footprint=disk(5)).astype(np.uint8)
    pus = ((L > L_thresh) & (b > B_thresh)).astype(np.uint8)
    combined = np.maximum(closed, pus)
    final = binary_opening(combined, footprint=disk(1)).astype(bool)
    final = binary_closing(final, footprint=disk(1)).astype(bool)
    final = remove_small_objects(final, min_size=200)
    final_filled = binary_fill_holes(final)
    final_clean = remove_small_objects(final_filled, min_size=200).astype(np.uint8) * 255
    roi = img_bgr.copy()
    roi[final_clean == 0] = 0

    return final_clean

# ============== Feature Extraction (must match training) ==============

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

# ============== Flask App ==============
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error':'No image file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error':'No selected file'}), 400

    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return jsonify({'error':'Could not decode image'}), 400

    img = cv2.resize(img, (150, 150))
    mask = segment_with_fcm(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feats = extract_features(img_gray, mask, img_rgb).reshape(1, -1)
    print("Extracted features shape:", feats.shape)
    print("Any NaN?", np.isnan(feats).any())
    print("Any Inf?", np.isinf(feats).any())

    feats_sel = selector.transform(feats)
    print("Selected features shape:", feats_sel.shape)
    feats_scaled = scaler.transform(feats_sel)
    print("Scaled features shape:", feats_scaled.shape)
    probs = elm.predict_proba(feats_scaled)
    print("Predicted probabilities:", probs)

    out = {c: float(round(p*100,2)) for c,p in zip(class_names, probs.ravel())}
    out['top'] = class_names[int(np.argmax(probs))]
    out['features'] = feats.flatten().tolist()

    overlay = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0,0,255), 2)
    _, buf = cv2.imencode('.png', overlay)
    out['overlay'] = f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"

    return jsonify(out)



if __name__ == '__main__':
    app.run(debug=True)
