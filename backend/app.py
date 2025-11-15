from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PARENT_DIR, "checkpoints", "best_effnet_b3.pth")
FALLBACK_MODEL = os.path.join(PARENT_DIR, "efficientnet_eye_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_FILTER = ["Cataract", "Conjunctivitis", "Normal"]

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Image preprocessing functions
def crop_eye_region_bgr(image_bgr):
    """Crop eye region from image"""
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as fm:
        res = fm.process(img_rgb)
        if not res.multi_face_landmarks:
            return image_bgr
        lm = res.multi_face_landmarks[0].landmark
        LEFT_EYE_IDX = [33, 133, 159, 145]
        RIGHT_EYE_IDX = [263, 362, 386, 374]
        def get_box(indices):
            xs = [int(lm[i].x * w) for i in indices]
            ys = [int(lm[i].y * h) for i in indices]
            x0, x1 = max(0,min(xs)), min(w, max(xs))
            y0, y1 = max(0,min(ys)), min(h, max(ys))
            pad = int(0.35 * max(y1-y0, x1-x0))
            return max(0,x0-pad), max(0,y0-pad), min(w,x1+pad), min(h,y1+pad)
        lx0, ly0, lx1, ly1 = get_box(LEFT_EYE_IDX)
        rx0, ry0, rx1, ry1 = get_box(RIGHT_EYE_IDX)
        left = image_bgr[ly0:ly1, lx0:lx1]
        right = image_bgr[ry0:ry1, rx0:rx1]
        if left.size==0 and right.size==0:
            return image_bgr
        if left.size >= right.size:
            return left if left.size>0 else right
        else:
            return right

def apply_clahe_rgb(image_bgr):
    """Apply CLAHE enhancement"""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

# Transforms
val_aug = A.Compose([
    A.Resize(300,300),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

tta_transforms = [
    A.Compose([A.Resize(300,300), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()]),
    A.Compose([A.HorizontalFlip(p=1.0), A.Resize(300,300), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()]),
    A.Compose([A.Rotate(limit=10, p=1.0), A.Resize(300,300), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
]

# GradCAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(image_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output).item()
        target_score = output[0, class_idx]
        target_score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

# Load model
print("Loading model...")
model = EfficientNet.from_pretrained('efficientnet-b3')
num_classes = len(CLASS_FILTER)
in_f = model._fc.in_features
model._fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_f, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

# Try loading from checkpoints first, then fallback to root
if os.path.exists(MODEL_PATH):
    print(f"Loading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
elif os.path.exists(FALLBACK_MODEL):
    print(f"Loading model from: {FALLBACK_MODEL}")
    checkpoint = torch.load(FALLBACK_MODEL, map_location=DEVICE, weights_only=False)
else:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH} or {FALLBACK_MODEL}")

model.load_state_dict(checkpoint['state_dict'])
model = model.to(DEVICE)
model.eval()

# Initialize GradCAM
target_layer = model._blocks[-1]._project_conv
cam = GradCAM(model, target_layer)

print("Model loaded successfully!")

def preprocess_image(image_bytes):
    """Preprocess uploaded image"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")
    
    # Crop eye region
    cropped = crop_eye_region_bgr(img_bgr)
    # Apply CLAHE
    enhanced = apply_clahe_rgb(cropped)
    # Convert to RGB
    img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return img_rgb

def predict_with_tta(img_rgb, model, tta_list=tta_transforms):
    """Predict with Test Time Augmentation"""
    probs_accum = np.zeros(len(CLASS_FILTER), dtype=np.float32)
    with torch.no_grad():
        for t in tta_list:
            inp = t(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
            out = model(inp)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
            probs_accum += probs
    probs_accum /= len(tta_list)
    return probs_accum

def generate_gradcam(img_rgb, model, cam):
    """Generate Grad-CAM visualization"""
    inp = val_aug(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
    heat = cam(inp)
    
    # Create heatmap
    heatmap = cv2.applyColorMap((heat * 255).astype("uint8"), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    
    # Overlay
    overlay = (0.55 * img_rgb + 0.45 * heatmap).astype("uint8")
    
    # Convert to base64
    pil_image = Image.fromarray(overlay)
    buff = io.BytesIO()
    pil_image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image_bytes = file.read()
        img_rgb = preprocess_image(image_bytes)
        
        # Predict
        probs = predict_with_tta(img_rgb, model)
        pred_idx = int(np.argmax(probs))
        prediction = CLASS_FILTER[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        
        # Generate Grad-CAM
        heatmap_image = generate_gradcam(img_rgb, model, cam)
        
        # Format predictions
        predictions = []
        for i, (cls, prob) in enumerate(zip(CLASS_FILTER, probs)):
            predictions.append({
                'label': cls,
                'name': cls,
                'confidence': float(prob * 100),
                'probability': float(prob),
                'severity': 'moderate' if prob > 0.5 else 'low',
                'threshold': 50.0,
                'detected': i == pred_idx
            })
        
        # Get top prediction
        top_prediction = predictions[pred_idx]
        
        # Get detected diseases (above threshold)
        detected_diseases = [p for p in predictions if p['confidence'] > 50.0]
        if not detected_diseases:
            detected_diseases = [top_prediction]
        
        # Generate recommendations
        recommendations = {
            'Cataract': 'Consult an ophthalmologist for cataract evaluation. Early detection can help manage symptoms effectively.',
            'Conjunctivitis': 'Seek medical attention. Conjunctivitis can be contagious and may require treatment.',
            'Normal': 'No significant abnormalities detected. Continue regular eye checkups.'
        }
        
        explanations = {
            'Cataract': 'Cataract is a clouding of the eye lens that can cause vision problems.',
            'Conjunctivitis': 'Conjunctivitis is inflammation of the conjunctiva, often causing redness and irritation.',
            'Normal': 'The eye appears healthy with no signs of disease.'
        }
        
        response = {
            'top_prediction': top_prediction,
            'predictions': predictions,
            'detected_diseases': detected_diseases,
            'recommendation': recommendations.get(prediction, 'Please consult a healthcare professional.'),
            'explanation': explanations.get(prediction, ''),
            'heatmap_image': heatmap_image,
            'all_probabilities': {cls: float(prob * 100) for cls, prob in zip(CLASS_FILTER, probs)}
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    print(f"Starting server on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)

