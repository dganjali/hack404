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

// In-memory storage (in production, use a database)
const users = [];
const userShelters = {}; // userId -> shelters array
const shelterData = {}; // shelterId -> historical data
const alerts = {}; // userId -> alerts array

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

// Helper function to generate shelter ID
const generateShelterId = () => {
  return 'shelter_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
};

// Helper function to generate alert ID
const generateAlertId = () => {
  return 'alert_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
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
    
    // Initialize user data
    userShelters[user.id] = [];
    alerts[user.id] = [];

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

    // Initialize user data if not exists
    if (!userShelters[user.id]) {
      userShelters[user.id] = [];
    }
    if (!alerts[user.id]) {
      alerts[user.id] = [];
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

// Get user's shelters
app.get('/api/shelters', authenticateToken, (req, res) => {
  try {
    const userSheltersList = userShelters[req.user.userId] || [];
    res.json({ shelters: userSheltersList });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Add new shelter
app.post('/api/shelters', authenticateToken, (req, res) => {
  try {
    const { name, address, maxCapacity, phone, email, description } = req.body;
    
    if (!name || !address || !maxCapacity) {
      return res.status(400).json({ error: 'Name, address, and max capacity are required' });
    }

    const shelter = {
      id: generateShelterId(),
      name,
      address,
      maxCapacity: parseInt(maxCapacity),
      phone: phone || '',
      email: email || '',
      description: description || '',
      createdAt: new Date().toISOString(),
      userId: req.user.userId
    };

    if (!userShelters[req.user.userId]) {
      userShelters[req.user.userId] = [];
    }
    
    userShelters[req.user.userId].push(shelter);
    shelterData[shelter.id] = [];

    res.status(201).json({
      message: 'Shelter added successfully',
      shelter
    });

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Update shelter
app.put('/api/shelters/:shelterId', authenticateToken, (req, res) => {
  try {
    const { shelterId } = req.params;
    const { name, address, maxCapacity, phone, email, description } = req.body;
    
    const userSheltersList = userShelters[req.user.userId] || [];
    const shelterIndex = userSheltersList.findIndex(s => s.id === shelterId);
    
    if (shelterIndex === -1) {
      return res.status(404).json({ error: 'Shelter not found' });
    }

    userSheltersList[shelterIndex] = {
      ...userSheltersList[shelterIndex],
      name: name || userSheltersList[shelterIndex].name,
      address: address || userSheltersList[shelterIndex].address,
      maxCapacity: maxCapacity ? parseInt(maxCapacity) : userSheltersList[shelterIndex].maxCapacity,
      phone: phone || userSheltersList[shelterIndex].phone,
      email: email || userSheltersList[shelterIndex].email,
      description: description || userSheltersList[shelterIndex].description,
      updatedAt: new Date().toISOString()
    };

    res.json({
      message: 'Shelter updated successfully',
      shelter: userSheltersList[shelterIndex]
    });

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Delete shelter
app.delete('/api/shelters/:shelterId', authenticateToken, (req, res) => {
  try {
    const { shelterId } = req.params;
    
    const userSheltersList = userShelters[req.user.userId] || [];
    const shelterIndex = userSheltersList.findIndex(s => s.id === shelterId);
    
    if (shelterIndex === -1) {
      return res.status(404).json({ error: 'Shelter not found' });
    }

    userSheltersList.splice(shelterIndex, 1);
    delete shelterData[shelterId];

    res.json({ message: 'Shelter deleted successfully' });

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Add real occupancy data
app.post('/api/shelters/:shelterId/data', authenticateToken, (req, res) => {
  try {
    const { shelterId } = req.params;
    const { date, occupancy, notes } = req.body;
    
    if (!date || occupancy === undefined) {
      return res.status(400).json({ error: 'Date and occupancy are required' });
    }

    const userSheltersList = userShelters[req.user.userId] || [];
    const shelter = userSheltersList.find(s => s.id === shelterId);
    
    if (!shelter) {
      return res.status(404).json({ error: 'Shelter not found' });
    }

    const dataPoint = {
      id: Date.now().toString(),
      date,
      occupancy: parseInt(occupancy),
      notes: notes || '',
      timestamp: new Date().toISOString()
    };

    if (!shelterData[shelterId]) {
      shelterData[shelterId] = [];
    }
    
    shelterData[shelterId].push(dataPoint);

    // Check for alerts
    const utilizationRate = (occupancy / shelter.maxCapacity) * 100;
    if (utilizationRate >= 90) {
      const alert = {
        id: generateAlertId(),
        type: 'high_occupancy',
        title: 'High Occupancy Alert',
        message: `${shelter.name} is at ${utilizationRate.toFixed(1)}% capacity`,
        severity: 'high',
        shelterId,
        shelterName: shelter.name,
        timestamp: new Date().toISOString(),
        read: false
      };
      
      if (!alerts[req.user.userId]) {
        alerts[req.user.userId] = [];
      }
      alerts[req.user.userId].push(alert);
    }

    res.status(201).json({
      message: 'Data added successfully',
      dataPoint
    });

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Get shelter data
app.get('/api/shelters/:shelterId/data', authenticateToken, (req, res) => {
  try {
    const { shelterId } = req.params;
    const { startDate, endDate } = req.query;
    
    const userSheltersList = userShelters[req.user.userId] || [];
    const shelter = userSheltersList.find(s => s.id === shelterId);
    
    if (!shelter) {
      return res.status(404).json({ error: 'Shelter not found' });
    }

    let data = shelterData[shelterId] || [];
    
    // Filter by date range if provided
    if (startDate && endDate) {
      data = data.filter(d => d.date >= startDate && d.date <= endDate);
    }

    res.json({ data });

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Predict occupancy for a shelter
app.post('/api/shelters/:shelterId/predict', authenticateToken, (req, res) => {
  try {
    const { shelterId } = req.params;
    const { date } = req.body;
    
    if (!date) {
      return res.status(400).json({ error: 'Date is required' });
    }

    const userSheltersList = userShelters[req.user.userId] || [];
    const shelter = userSheltersList.find(s => s.id === shelterId);
    
    if (!shelter) {
      return res.status(404).json({ error: 'Shelter not found' });
    }

    // For now, we'll simulate predictions based on historical data
    // In production, you'd call the Python prediction model
    const historicalData = shelterData[shelterId] || [];
    let prediction = Math.floor(Math.random() * shelter.maxCapacity * 0.8) + Math.floor(shelter.maxCapacity * 0.2);
    
    if (historicalData.length > 0) {
      // Use average of recent data as base prediction
      const recentData = historicalData.slice(-7); // Last 7 days
      const avgOccupancy = recentData.reduce((sum, d) => sum + d.occupancy, 0) / recentData.length;
      prediction = Math.floor(avgOccupancy * (0.8 + Math.random() * 0.4)); // Add some variation
    }

    res.json({
      shelterId,
      shelterName: shelter.name,
      date,
      predicted_occupancy: prediction,
      maxCapacity: shelter.maxCapacity
    });

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Get user's alerts
app.get('/api/alerts', authenticateToken, (req, res) => {
  try {
    const userAlerts = alerts[req.user.userId] || [];
    res.json({ alerts: userAlerts });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Mark alert as read
app.put('/api/alerts/:alertId/read', authenticateToken, (req, res) => {
  try {
    const { alertId } = req.params;
    const userAlerts = alerts[req.user.userId] || [];
    const alertIndex = userAlerts.findIndex(a => a.id === alertId);
    
    if (alertIndex === -1) {
      return res.status(404).json({ error: 'Alert not found' });
    }

    userAlerts[alertIndex].read = true;
    res.json({ message: 'Alert marked as read' });

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Get dashboard data
app.get('/api/dashboard', authenticateToken, (req, res) => {
  try {
    const userSheltersList = userShelters[req.user.userId] || [];
    const userAlerts = alerts[req.user.userId] || [];
    const today = new Date().toISOString().split('T')[0];
    
    let totalPredicted = 0;
    let totalCapacity = 0;
    let totalUtilization = 0;
    let shelterCount = 0;

    const dashboardData = {
      date: today,
      total_predicted: 0,
      total_capacity: 0,
      utilization_rate: 0,
      alerts_count: userAlerts.filter(a => !a.read).length,
      shelters: []
    };

    // Calculate totals and prepare shelter data
    userSheltersList.forEach(shelter => {
      const shelterDataList = shelterData[shelter.id] || [];
      const todayData = shelterDataList.find(d => d.date === today);
      const currentOccupancy = todayData ? todayData.occupancy : 0;
      
      // Simulate prediction for today
      const prediction = Math.floor(Math.random() * shelter.maxCapacity * 0.8) + Math.floor(shelter.maxCapacity * 0.2);
      
      totalPredicted += prediction;
      totalCapacity += shelter.maxCapacity;
      totalUtilization += currentOccupancy;
      shelterCount++;

      dashboardData.shelters.push({
        id: shelter.id,
        name: shelter.name,
        address: shelter.address,
        maxCapacity: shelter.maxCapacity,
        currentOccupancy,
        predicted_occupancy: prediction,
        utilization_rate: Math.round((currentOccupancy / shelter.maxCapacity) * 100)
      });
    });

    dashboardData.total_predicted = totalPredicted;
    dashboardData.total_capacity = totalCapacity;
    dashboardData.utilization_rate = shelterCount > 0 ? Math.round((totalUtilization / totalCapacity) * 100) : 0;

    res.json(dashboardData);

  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 