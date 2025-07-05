#!/bin/bash

# Start the webapp with new location-based prediction parameters

echo "Starting ShelterFlow Webapp with Location-Based Predictions"
echo "=========================================================="

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 not found. Please install it first."
    echo "You can use pyenv: pyenv install 3.11.0"
    exit 1
fi

# Check if required Python packages are installed
echo "📦 Checking Python dependencies..."
cd backend/prediction
python3.11 -c "import tensorflow, pandas, numpy, joblib, holidays" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing Python dependencies. Installing..."
    pip3.11 install -r requirements.txt
fi

# Check if the trained model exists
if [ ! -f "models/best_model.h5" ]; then
    echo "❌ Trained model not found. Please ensure best_model.h5 exists in backend/prediction/models/"
    echo "You can copy it from the root models/ directory:"
    echo "cp ../../models/best_model.h5 models/"
    exit 1
fi

# Start Python prediction API
echo "🚀 Starting Python Prediction API..."
python3.11 api.py &
PYTHON_PID=$!

# Wait a moment for Python API to start
sleep 3

# Check if Python API is running
if ! curl -s http://localhost:5000/health > /dev/null; then
    echo "❌ Python API failed to start"
    kill $PYTHON_PID 2>/dev/null
    exit 1
fi

echo "✅ Python Prediction API started successfully"

# Install Node.js dependencies if needed
echo "📦 Checking Node.js dependencies..."
cd ../server
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

# Start Node.js server
echo "🚀 Starting Node.js Webapp Server..."
npm start &
NODE_PID=$!

# Wait a moment for Node.js server to start
sleep 3

# Check if Node.js server is running
if ! curl -s http://localhost:3000 > /dev/null; then
    echo "❌ Node.js server failed to start"
    kill $PYTHON_PID $NODE_PID 2>/dev/null
    exit 1
fi

echo "✅ Node.js Webapp Server started successfully"
echo ""
echo "🎉 ShelterFlow Webapp is now running!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Python API: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $PYTHON_PID $NODE_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for background processes
wait 