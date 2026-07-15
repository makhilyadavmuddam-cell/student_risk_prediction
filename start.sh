#!/bin/bash
# EduSense — Student Risk Intelligence Platform
# Run this from the project root directory

echo "=================================="
echo "  EduSense Platform — Starting"
echo "=================================="

# Step 1: Generate data
echo ""
echo "[1/3] Generating student dataset..."
python3 backend/data/generate_data.py

# Step 2: Train model
echo ""
echo "[2/3] Training ML model..."
python3 backend/model/train.py

# Step 3: Start FastAPI
echo ""
echo "[3/3] Starting API server at http://localhost:8000"
echo "      Open frontend/index.html in your browser"
echo ""
cd backend
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
