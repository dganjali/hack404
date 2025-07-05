# ShelterFlow - AI-Powered Shelter Management

A comprehensive web application that uses machine learning to predict shelter occupancy and optimize resource allocation for homeless shelters in Toronto.

## 🌟 Features

### 🔍 Problem Solved
- **Fluctuating Demand**: Unpredictable bed usage across locations
- **Poor Coordination**: Limited real-time communication between shelters  
- **Resource Waste**: Unused beds in one place while others overflow
- **No Data-Driven Planning**: Lack of centralized tools for informed decisions

### 💡 Solution
- **AI-Powered Forecasting**: LSTM neural networks with 94.5% accuracy
- **Resource Optimization**: Smart algorithms for bed, staff, and supply allocation
- **Real-Time Dashboard**: Live monitoring with alerts and recommendations

## 🏗️ Architecture

```
hack404/
├── frontend/                    # HTML/CSS/JS frontend
│   ├── html/
│   │   ├── index.html          # Home page with product info
│   │   └── dashboard.html      # Dashboard interface
│   ├── css/
│   │   ├── style.css          # Main styling
│   │   └── dashboard.css      # Dashboard styling
│   └── js/
│       ├── auth.js            # Authentication logic
│       ├── main.js            # Home page functionality
│       └── dashboard.js       # Dashboard functionality
├── backend/
│   ├── server/                # Node.js backend
│   │   ├── server.js         # Express server with auth
│   │   └── package.json      # Node.js dependencies
│   └── prediction/           # Python ML prediction system
│       ├── models/           # Trained models & parameters
│       ├── shelter_predictor.py  # Core prediction class
│       ├── api.py            # Flask API (optional)
│       └── requirements.txt  # Python dependencies
├── data/                     # Historical shelter data
├── package.json              # Main project configuration
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites
- Node.js 16+
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hack404
   ```

2. **Install dependencies**
   ```bash
   npm run install-deps
   ```

3. **Start the application**
   ```bash
   npm start
   ```

4. **Access the application**
   - Home page: http://localhost:3000
   - Dashboard: http://localhost:3000/dashboard (after login)

## 📊 Model Performance

- **MAE**: 6.98 people
- **RMSE**: 19.81 people
- **R² Score**: 0.9454
- **Training Data**: 91,216 sequences
- **Test Data**: 22,804 sequences

## 🎯 Core Functionalities

### 1. Occupancy Forecasting
- LSTM neural networks analyze historical patterns
- Predicts daily bed usage with high accuracy
- Considers weather, seasonality, and temporal patterns

### 2. Resource Balancing Engine
- Linear programming optimization
- Reallocates beds, food, staff across shelters
- Based on predicted demand and current capacity

### 3. Live Dashboard
- Real-time occupancy visualization
- Urgent alerts and notifications
- Resource allocation recommendations
- Interactive shelter management

## 🔐 Authentication

The application includes:
- User registration and login
- JWT token-based authentication
- Secure password hashing with bcrypt
- Protected API endpoints

## 🎨 User Interface

### Home Page
- Product description and problem statement
- Interactive statistics and animations
- Sign up/Sign in modals
- Responsive design

### Dashboard
- Overview with key metrics
- Individual shelter predictions
- Resource management interface
- Alerts and notifications system

## 🚀 Deployment

### Render Deployment
The application is configured for easy deployment on Render:

**Build Command:**
```bash
npm install
```

**Start Command:**
```bash
npm start
```

### Environment Variables
Create a `.env` file in the root directory:
```
JWT_SECRET=your-secret-key-here
PORT=3000
```

## 🔧 API Endpoints

### Authentication
- `POST /api/register` - User registration
- `POST /api/login` - User login

### Dashboard
- `GET /api/dashboard` - Get dashboard data
- `GET /api/shelters` - Get available shelters
- `POST /api/predict` - Make occupancy prediction

## 📈 Technology Stack

### Frontend
- **HTML5/CSS3**: Modern, responsive design
- **JavaScript**: Interactive functionality
- **Font Awesome**: Icons and UI elements
- **Google Fonts**: Typography

### Backend
- **Node.js**: Server runtime
- **Express.js**: Web framework
- **JWT**: Authentication
- **bcryptjs**: Password hashing
- **python-shell**: Python integration

### Machine Learning
- **TensorFlow**: Deep learning framework
- **LSTM**: Neural network architecture
- **scikit-learn**: Data preprocessing
- **pandas/numpy**: Data manipulation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Toronto Open Data for shelter occupancy data
- TensorFlow team for the deep learning framework
- The homeless shelter community for inspiration

---

**Built with ❤️ for better shelter management** 