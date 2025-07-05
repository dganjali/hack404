const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const cors = require('cors');
const path = require('path');
const { PythonShell } = require('python-shell');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../../frontend')));

// In-memory user storage (in production, use a database)
const users = [];

// JWT Secret (in production, use environment variable)
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ error: 'Invalid token' });
    }
    req.user = user;
    next();
  });
};

// Routes

// Serve home page
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../../frontend/html/index.html'));
});

// Serve dashboard page
app.get('/dashboard', (req, res) => {
  res.sendFile(path.join(__dirname, '../../frontend/html/dashboard.html'));
});

// Register endpoint
app.post('/api/register', async (req, res) => {
  try {
    const { username, email, password } = req.body;

    // Check if user already exists
    if (users.find(user => user.email === email)) {
      return res.status(400).json({ error: 'User already exists' });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create user
    const user = {
      id: users.length + 1,
      username,
      email,
      password: hashedPassword
    };

    users.push(user);

    // Generate JWT token
    const token = jwt.sign({ userId: user.id, email: user.email }, JWT_SECRET);

    res.status(201).json({
      message: 'User registered successfully',
      token,
      user: { id: user.id, username: user.username, email: user.email }
    });

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Login endpoint
app.post('/api/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find user
    const user = users.find(u => u.email === email);
    if (!user) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }

    // Check password
    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }

    // Generate JWT token
    const token = jwt.sign({ userId: user.id, email: user.email }, JWT_SECRET);

    res.json({
      message: 'Login successful',
      token,
      user: { id: user.id, username: user.username, email: user.email }
    });

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Get available shelters
app.get('/api/shelters', authenticateToken, (req, res) => {
  try {
    const options = {
      mode: 'text',
      pythonPath: 'python3',
      pythonOptions: ['-u'],
      scriptPath: path.join(__dirname, '../prediction'),
      args: ['--list-shelters']
    };

    PythonShell.run('shelter_predictor.py', options, (err, results) => {
      if (err) {
        console.error('Error getting shelters:', err);
        return res.status(500).json({ error: 'Failed to get shelters' });
      }

      // Parse the results to extract shelter names
      const shelters = [];
      if (results && results.length > 0) {
        // The Python script should return shelter names
        // For now, we'll return a default list
        shelters.push(
          "COSTI Reception Centre",
          "Christie Ossington Men's Hostel",
          "Christie Refugee Welcome Centre",
          "Birchmount Residence",
          "Birkdale Residence",
          "Downsview Dells",
          "Family Residence"
        );
      }

      res.json({ shelters });
    });

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Predict occupancy
app.post('/api/predict', authenticateToken, (req, res) => {
  try {
    const { date, shelter_name } = req.body;

    if (!date || !shelter_name) {
      return res.status(400).json({ error: 'Date and shelter name are required' });
    }

    const options = {
      mode: 'text',
      pythonPath: 'python3',
      pythonOptions: ['-u'],
      scriptPath: path.join(__dirname, '../prediction'),
      args: [date, shelter_name]
    };

    PythonShell.run('predict_single.py', options, (err, results) => {
      if (err) {
        console.error('Error making prediction:', err);
        return res.status(500).json({ error: 'Failed to make prediction' });
      }

      // Parse the prediction result
      let prediction = 0;
      if (results && results.length > 0) {
        prediction = parseInt(results[0]) || 0;
      }

      res.json({
        date,
        shelter_name,
        predicted_occupancy: prediction
      });
    });

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Get dashboard data
app.get('/api/dashboard', authenticateToken, (req, res) => {
  try {
    const today = new Date().toISOString().split('T')[0];
    
    // Get predictions for all shelters for today
    const shelters = [
      "COSTI Reception Centre",
      "Christie Ossington Men's Hostel",
      "Christie Refugee Welcome Centre",
      "Birchmount Residence",
      "Birkdale Residence",
      "Downsview Dells",
      "Family Residence"
    ];

    const dashboardData = {
      date: today,
      total_predicted: 0,
      shelters: []
    };

    // For now, we'll simulate predictions
    // In production, you'd call the Python prediction for each shelter
    shelters.forEach((shelter, index) => {
      const prediction = Math.floor(Math.random() * 50) + 10; // Simulated prediction
      dashboardData.total_predicted += prediction;
      dashboardData.shelters.push({
        name: shelter,
        predicted_occupancy: prediction,
        capacity: 100, // This would come from your data
        utilization_rate: Math.round((prediction / 100) * 100)
      });
    });

    res.json(dashboardData);

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 