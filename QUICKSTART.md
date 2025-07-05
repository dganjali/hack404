# 🚀 ProcessUnravel Pro - Quick Start Guide

## Prerequisites

- **Python 3.8+** - For the backend AI agents and API
- **Node.js 16+** - For the React frontend
- **OpenAI API Key** - For AI processing (get one at https://platform.openai.com/)

## Installation

### Option 1: Automated Setup (Recommended)
```bash
# Make the setup script executable and run it
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

#### Backend Setup
```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install

# Create environment file
cp .env.example .env
# Edit .env and add your OpenAI API key
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install
```

## Configuration

### 1. Add OpenAI API Key
Edit `backend/.env` and add your OpenAI API key:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 2. Optional: Install Tesseract OCR
For PDF processing with OCR:
- **macOS**: `brew install tesseract`
- **Ubuntu**: `sudo apt install tesseract-ocr`
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki

## Running the Application

### Start Backend
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```
Backend will be available at: http://localhost:8000

### Start Frontend
```bash
cd frontend
npm start
```
Frontend will be available at: http://localhost:3000

## Testing the System

### 1. Create Sample Process
1. Go to http://localhost:3000
2. Click "Create Your Process"
3. Click "Create Sample Process" to see the food truck example

### 2. View Interactive Process
1. After creating the sample, click "View Process"
2. Explore the interactive decision tree
3. Click on nodes to see detailed information
4. Use the personalization panel to filter steps

### 3. Scrape Real Government Sites
1. Go to "Create New Process"
2. Add government website URLs
3. Provide a process name
4. Click "Create Process"

## API Documentation

Once the backend is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## Key Features to Test

### 🧠 AI Agents
- **PageChunkerAgent**: Breaks content into logical chunks
- **StepClassifierAgent**: Classifies steps (ACTION, DOCUMENT, FEE, etc.)
- **DependencyAgent**: Identifies step dependencies

### 🌐 Web Scraping
- Scrapes government websites using Playwright
- Extracts main content and metadata
- Handles multiple URLs concurrently

### 📄 PDF Processing
- Extracts text from government PDFs
- OCR support for scanned documents
- Form and table detection

### 🎯 Personalization
- Filter steps based on user context
- Province/city-specific requirements
- Business type and industry filtering

### 📊 Interactive Visualization
- React Flow-based decision trees
- Click-to-expand node details
- Color-coded step types

## Troubleshooting

### Common Issues

**1. OpenAI API Key Error**
```
Error: No API key provided
```
Solution: Add your OpenAI API key to `backend/.env`

**2. Playwright Browser Error**
```
Error: Browser not found
```
Solution: Run `playwright install` in the backend directory

**3. Frontend Build Error**
```
Error: Cannot find module 'react'
```
Solution: Run `npm install` in the frontend directory

**4. Port Already in Use**
```
Error: Address already in use
```
Solution: Change ports in `backend/main.py` or `frontend/package.json`

### Getting Help

- Check the console logs for detailed error messages
- Ensure all dependencies are installed
- Verify your OpenAI API key is valid
- Make sure ports 3000 and 8000 are available

## Next Steps

1. **Add More Processes**: Scrape different government websites
2. **Customize AI Agents**: Modify prompts in `backend/agents/`
3. **Enhance UI**: Add more React components
4. **Add Database**: Store processes persistently
5. **Deploy**: Use Vercel for frontend, Render for backend

## Sample Use Cases

### Food Truck Business
- **URLs**: Ontario business registration, Toronto mobile vending
- **Steps**: 8 steps with dependencies
- **Personalization**: Location, business type, industry

### Business Registration
- **URLs**: Provincial business registry, CRA business number
- **Steps**: 5 steps with document requirements
- **Personalization**: Province, incorporation status

### Building Permits
- **URLs**: Municipal building department, provincial codes
- **Steps**: 12 steps with inspections
- **Personalization**: City, project type, building codes 