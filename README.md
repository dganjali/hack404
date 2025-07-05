# NeedsMatcher MVP

A smart, interactive system that forecasts tomorrow's shelter demand, optimizes resource allocation between shelters, and visualizes it all in a clean dashboard.

## 🎯 Features

- **Demand Forecasting**: ML-powered prediction of tomorrow's shelter needs
- **Resource Optimization**: Linear programming to minimize shortages across shelters
- **Interactive Dashboard**: Real-time visualization of forecasts and transfer plans
- **Smart Transfers**: Automated recommendations for resource redistribution

## 🚀 Quick Start

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Generate Mock Data
```bash
cd data
python generate_mock_data.py
```

## 📁 Project Structure

```
needsmatcher/
├── backend/           # FastAPI + ML models
├── frontend/          # React + Tailwind dashboard
├── data/              # Mock data generation
└── README.md
```

## 🔧 Tech Stack

- **Backend**: FastAPI, scikit-learn, PuLP
- **Frontend**: React, Tailwind CSS, Chart.js
- **ML**: RandomForest for demand forecasting
- **Optimization**: Linear programming for resource allocation