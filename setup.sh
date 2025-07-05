#!/bin/bash

echo "🚀 Setting up NeedsMatcher MVP with Toronto Data..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

echo "📦 Installing backend dependencies..."
cd backend
python3 -m pip install -r requirements.txt
cd ..

echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "🗃️ Processing Toronto shelter data..."
cd data
python3 process_toronto_data.py
cd ..

echo "✅ Setup complete!"
echo ""
echo "🎯 To start the application:"
echo "   1. Start backend: cd backend && uvicorn main:app --reload"
echo "   2. Start frontend: cd frontend && npm start"
echo "   3. Open http://localhost:3000"
echo ""
echo "📊 The dashboard will now show:"
echo "   • Real Toronto shelter data (38,528 actively homeless)"
echo "   • Enhanced ML predictions with demographic features"
echo "   • Toronto-specific analytics and insights"
echo "   • 73.56% housing success rate tracking"
echo "   • Interactive charts with real data patterns" 