# 🚀 Quick Start Guide

## Step 1: Start the Backend Server

Open a terminal and run:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Wait for: `Starting server on port 5001...`

## Step 2: Start the Frontend

Open a **NEW terminal** and run:

```bash
cd frontend
npm install
npm run dev
```

## Step 3: Open the Application

Open your browser and go to the URL shown in the frontend terminal (usually `http://localhost:3000`)

## Step 4: Upload and Scan

1. Click "Upload Image" button
2. Select an image from your device
3. The image will automatically be sent to the backend
4. View the results with Grad-CAM visualization!

## Troubleshooting

- **Backend won't start:** Make sure you have the model file at `checkpoints/best_effnet_b3.pth`
- **Frontend can't connect:** Make sure backend is running on port 5001
- **Port conflicts:** Change port in `backend/app.py` (line with `port=5001`)

## That's it! 🎉

Your complete eye disease classification application is ready!

