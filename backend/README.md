# Eye Disease Classification Backend API

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the trained model:
   - `./checkpoints/best_effnet_b3.pth` OR
   - `./efficientnet_eye_model.pth`

3. Run the server:
```bash
python app.py
```

The API will run on `http://localhost:5001`

## API Endpoints

### POST /predict
Upload an image file to get predictions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
  "top_prediction": {
    "label": "Cataract",
    "name": "Cataract",
    "confidence": 85.5,
    "probability": 0.855,
    "severity": "moderate",
    "threshold": 50.0,
    "detected": true
  },
  "predictions": [...],
  "detected_diseases": [...],
  "recommendation": "...",
  "explanation": "...",
  "heatmap_image": "data:image/png;base64,...",
  "all_probabilities": {...}
}
```

### GET /health
Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

