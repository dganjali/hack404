const express = require('express');
const axios = require('axios');
const { auth } = require('../middleware/auth');

const router = express.Router();

// ML service URL
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5000';

// Get prediction for a specific shelter
router.get('/shelter/:id', auth, async (req, res) => {
  try {
    const { id } = req.params;
    const { days = '7' } = req.query;
    
    // Call ML service for prediction
    const response = await axios.get(`${ML_SERVICE_URL}/predict/shelter/${id}`, {
      params: { days },
      timeout: 10000
    });
    
    res.json(response.data);
  } catch (error) {
    console.error('Prediction error:', error.message);
    res.status(500).json({ 
      error: 'Failed to get prediction',
      message: 'ML service unavailable'
    });
  }
});

// Get predictions for all shelters
router.get('/overview', auth, async (req, res) => {
  try {
    const { days = '7' } = req.query;
    
    // Call ML service for overview predictions
    const response = await axios.get(`${ML_SERVICE_URL}/predict/overview`, {
      params: { days },
      timeout: 15000
    });
    
    res.json(response.data);
  } catch (error) {
    console.error('Overview prediction error:', error.message);
    res.status(500).json({ 
      error: 'Failed to get overview predictions',
      message: 'ML service unavailable'
    });
  }
});

// Get resource predictions
router.get('/resources/:id', auth, async (req, res) => {
  try {
    const { id } = req.params;
    const { days = '7' } = req.query;
    
    // Call ML service for resource predictions
    const response = await axios.get(`${ML_SERVICE_URL}/predict/resources/${id}`, {
      params: { days },
      timeout: 10000
    });
    
    res.json(response.data);
  } catch (error) {
    console.error('Resource prediction error:', error.message);
    res.status(500).json({ 
      error: 'Failed to get resource predictions',
      message: 'ML service unavailable'
    });
  }
});

// Get model status
router.get('/status', auth, async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/health`, {
      timeout: 5000
    });
    
    res.json({
      status: 'healthy',
      ml_service: response.data
    });
  } catch (error) {
    res.json({
      status: 'unhealthy',
      ml_service: 'unavailable',
      error: error.message
    });
  }
});

module.exports = router; 