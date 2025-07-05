# Shelter Management Dashboard

A comprehensive web application for homeless shelter management with AI-powered predictions for occupancy and resource planning.

## Features

- **Real-time Analytics**: Monitor shelter occupancy, utilization rates, and trends
- **AI Predictions**: ML-powered forecasts for future occupancy and resource needs  
- **Resource Planning**: Optimize meals, kits, and staff allocation
- **User Management**: Secure authentication with role-based access
- **Modern UI**: Clean, responsive dashboard built with HTML/CSS/JS

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Backend**: Node.js, Express, MongoDB
- **ML Service**: Python, FastAPI, PyTorch
- **Charts**: Chart.js
- **Authentication**: JWT tokens

## Project Structure

```
hack404/
├── data/                   # Shelter data files
├── ml_service/            # Python ML service
│   ├── app.py            # FastAPI application
│   ├── model.py          # PyTorch neural network
│   ├── train.py          # Training script
│   └── requirements.txt  # Python dependencies
├── models/               # MongoDB schemas
├── routes/              # Express API routes
├── middleware/          # Authentication middleware
├── public/              # Static frontend files
│   ├── css/
│   ├── js/
│   └── index.html
├── server.js            # Express server
├── package.json         # Node.js dependencies
└── .env.example         # Environment variables template
```

## Quick Start

### Prerequisites

- Node.js 16+
- Python 3.8+
- MongoDB (local or cloud)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hack404
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

3. **Install Python dependencies**
   ```bash
   cd ml_service
   pip install -r requirements.txt
   cd ..
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start the application**
   ```bash
   npm start
   ```

### Environment Variables

Create a `.env` file in the root directory:

```env
# Node.js backend
PORT=3000
MONGODB_URI=mongodb://localhost:27017/shelter_dashboard
JWT_SECRET=your_jwt_secret_here
NODE_ENV=development
ML_SERVICE_URL=http://localhost:5000
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user

### Shelters
- `GET /api/shelters` - Get all shelters
- `GET /api/shelters/:id` - Get shelter by ID
- `GET /api/shelters/:id/stats` - Get shelter statistics
- `GET /api/shelters/stats/overview` - Get overview statistics

### Predictions
- `GET /api/predictions/shelter/:id` - Get shelter predictions
- `GET /api/predictions/overview` - Get overview predictions
- `GET /api/predictions/resources/:id` - Get resource predictions

## ML Service

The ML service runs on port 5000 and provides:

- **Model Training**: Custom PyTorch LSTM model
- **Predictions**: Occupancy and resource forecasting
- **API**: FastAPI endpoints for predictions

### Training the Model

```bash
cd ml_service
python train.py --epochs 100 --learning-rate 0.001
```

### Running ML Service

```bash
cd ml_service
python app.py
```

## Deployment

### Render Deployment

1. **Connect your repository to Render**
2. **Set build command**: `npm install`
3. **Set start command**: `npm start`
4. **Add environment variables**:
   - `MONGODB_URI`
   - `JWT_SECRET`
   - `NODE_ENV=production`

### ML Service Deployment

Deploy the ML service separately on Render:

1. **Create a new Web Service**
2. **Set build command**: `pip install -r requirements.txt`
3. **Set start command**: `python app.py`
4. **Set environment variables**:
   - `PORT=5000`

## Data

The application uses real shelter occupancy data from 2017-2020, including:

- 66 shelters
- 3.6M+ occupancy records
- 92.6% average utilization rate

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License 