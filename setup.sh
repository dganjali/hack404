#!/bin/bash

echo "🚀 Setting up ProcessUnravel Pro..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed. Please install Node.js 16+ and try again."
    exit 1
fi

echo "✅ Python and Node.js found"

# Backend setup
echo "📦 Setting up backend..."
cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install Playwright browsers
echo "Installing Playwright browsers..."
playwright install

echo "✅ Backend setup complete"

# Frontend setup
echo "📦 Setting up frontend..."
cd ../frontend

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete"

# Create .env file for backend
cd ../backend
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# ProcessUnravel Pro Environment Variables
OPENAI_API_KEY=your_openai_api_key_here
ENVIRONMENT=development
DEBUG=True
EOF
    echo "⚠️  Please edit backend/.env and add your OpenAI API key"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "1. Backend: cd backend && source venv/bin/activate && python main.py"
echo "2. Frontend: cd frontend && npm start"
echo ""
echo "The application will be available at:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo ""
echo "Don't forget to:"
echo "- Add your OpenAI API key to backend/.env"
echo "- Install Tesseract OCR for PDF processing (optional)" 