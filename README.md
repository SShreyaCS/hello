# Eye Disease Classification Application

A complete web application for classifying eye diseases (Cataract, Conjunctivitis, Normal) using deep learning with Grad-CAM visualization.

## Project Structure

```
generic/
├── backend/              # Flask API server
│   ├── app.py           # Main API application
│   ├── requirements.txt # Backend dependencies
│   └── README.md        # Backend documentation
├── frontend/            # React/TypeScript frontend
│   ├── src/
│   │   ├── pages/
│   │   │   └── ProfessionalMode.tsx  # Main page
│   │   └── components/
│   │       └── UploadBox.tsx        # File upload component
│   └── package.json
├── checkpoints/         # Trained model checkpoints
│   └── best_effnet_b3.pth
├── dataset/            # Training dataset
│   ├── Cataract/
│   ├── Conjunctivitis/
│   └── Normal/
├── generic.ipynb       # Model training notebook
└── README.md          # This file
```

## Quick Start

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the server (runs on port 5001)
python app.py
```

The backend will start on `http://localhost:5001`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will start on `http://localhost:3000`

### 3. Using the Application

1. Open the frontend in your browser
2. Click "Upload Image" or drag & drop an image
3. The image will be automatically sent to the backend
4. View predictions and Grad-CAM visualization

## Features

- ✅ Image upload from device
- ✅ Real-time disease classification
- ✅ Grad-CAM heatmap visualization
- ✅ Confidence scores for all classes
- ✅ Professional recommendations
- ✅ Modern, responsive UI

## API Endpoints

### POST /predict
Upload an image for classification.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "top_prediction": {...},
  "predictions": [...],
  "detected_diseases": [...],
  "recommendation": "...",
  "explanation": "...",
  "heatmap_image": "data:image/png;base64,..."
}
```

### GET /health
Check API health status.

## Model Information

- **Architecture:** EfficientNet-B3
- **Classes:** Cataract, Conjunctivitis, Normal
- **Input Size:** 300x300
- **Features:** Test Time Augmentation (TTA), Grad-CAM visualization

## Requirements

### Backend
- Python 3.8+
- PyTorch
- Flask
- MediaPipe
- OpenCV
- Albumentations

### Frontend
- Node.js 16+
- React 18+
- TypeScript
- Vite

## Troubleshooting

1. **Model not found error:**
   - Ensure `checkpoints/best_effnet_b3.pth` exists
   - Or `efficientnet_eye_model.pth` in root directory

2. **Port already in use:**
   - Backend uses port 5001 (change in `backend/app.py`)
   - Frontend uses port 3000 (change in `frontend/vite.config.ts`)

3. **CORS errors:**
   - Backend has CORS enabled for all origins
   - Check if backend is running on correct port

## License

This project is for educational purposes.

