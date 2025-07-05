const express = require('express');
const fs = require('fs').promises;
const path = require('path');
const { auth, requireRole } = require('../middleware/auth');

const router = express.Router();

// Get all shelters
router.get('/', auth, async (req, res) => {
  try {
    const sheltersPath = path.join(__dirname, '../data/real_shelters.json');
    const sheltersData = await fs.readFile(sheltersPath, 'utf8');
    const shelters = JSON.parse(sheltersData);
    
    res.json({ shelters });
  } catch (error) {
    res.status(500).json({ error: 'Failed to load shelters data' });
  }
});

// Get shelter by ID
router.get('/:id', auth, async (req, res) => {
  try {
    const { id } = req.params;
    const sheltersPath = path.join(__dirname, '../data/real_shelters.json');
    const sheltersData = await fs.readFile(sheltersPath, 'utf8');
    const shelters = JSON.parse(sheltersData);
    
    const shelter = shelters.find(s => s.id === id);
    if (!shelter) {
      return res.status(404).json({ error: 'Shelter not found' });
    }
    
    res.json({ shelter });
  } catch (error) {
    res.status(500).json({ error: 'Failed to load shelter data' });
  }
});

// Get shelter statistics
router.get('/:id/stats', auth, async (req, res) => {
  try {
    const { id } = req.params;
    const { period = '30' } = req.query; // days
    
    // Load shelter data
    const sheltersPath = path.join(__dirname, '../data/real_shelters.json');
    const sheltersData = await fs.readFile(sheltersPath, 'utf8');
    const shelters = JSON.parse(sheltersData);
    
    const shelter = shelters.find(s => s.id === id);
    if (!shelter) {
      return res.status(404).json({ error: 'Shelter not found' });
    }
    
    // Load historical data
    const historyPath = path.join(__dirname, '../data/real_intake_history.json');
    const historyData = await fs.readFile(historyPath, 'utf8');
    const history = JSON.parse(historyData);
    
    // Filter history for this shelter
    const shelterHistory = history.filter(record => record.shelter_id === id);
    
    // Calculate statistics
    const stats = {
      shelter: shelter,
      totalRecords: shelterHistory.length,
      avgOccupancy: shelter.avg_occupancy,
      maxOccupancy: shelter.max_occupancy,
      utilizationRate: shelter.utilization_rate,
      capacity: shelter.capacity,
      currentBeds: shelter.current_beds,
      currentMeals: shelter.current_meals,
      currentKits: shelter.current_kits
    };
    
    res.json({ stats });
  } catch (error) {
    res.status(500).json({ error: 'Failed to load shelter statistics' });
  }
});

// Get shelter occupancy trends
router.get('/:id/trends', auth, async (req, res) => {
  try {
    const { id } = req.params;
    const { days = '30' } = req.query;
    
    // Load features data for trends
    const featuresPath = path.join(__dirname, '../data/real_features.json');
    const featuresData = await fs.readFile(featuresPath, 'utf8');
    const features = JSON.parse(featuresData);
    
    // Get recent data (last N days)
    const recentFeatures = features.slice(-parseInt(days));
    
    const trends = {
      dates: recentFeatures.map(f => f.date),
      totalOccupancy: recentFeatures.map(f => f.total_occupancy),
      avgOccupancy: recentFeatures.map(f => f.avg_occupancy),
      utilizationRate: recentFeatures.map(f => f.utilization_rate),
      shelterCount: recentFeatures.map(f => f.shelter_count)
    };
    
    res.json({ trends });
  } catch (error) {
    res.status(500).json({ error: 'Failed to load trend data' });
  }
});

// Get overall dashboard statistics
router.get('/stats/overview', auth, async (req, res) => {
  try {
    // Load summary data
    const summaryPath = path.join(__dirname, '../data/real_data_summary.json');
    const summaryData = await fs.readFile(summaryPath, 'utf8');
    const summary = JSON.parse(summaryData);
    
    // Load shelters data
    const sheltersPath = path.join(__dirname, '../data/real_shelters.json');
    const sheltersData = await fs.readFile(sheltersPath, 'utf8');
    const shelters = JSON.parse(sheltersData);
    
    const overview = {
      totalShelters: summary.shelters_count,
      totalRecords: summary.history_records,
      totalOccupancy: summary.total_occupancy,
      avgUtilization: summary.avg_utilization,
      dateRange: summary.date_range,
      sheltersBySector: {},
      topShelters: []
    };
    
    // Calculate shelters by sector
    shelters.forEach(shelter => {
      const sector = shelter.sector;
      if (!overview.sheltersBySector[sector]) {
        overview.sheltersBySector[sector] = 0;
      }
      overview.sheltersBySector[sector]++;
    });
    
    // Get top 5 shelters by utilization
    overview.topShelters = shelters
      .filter(s => s.utilization_rate > 0)
      .sort((a, b) => b.utilization_rate - a.utilization_rate)
      .slice(0, 5)
      .map(s => ({
        id: s.id,
        name: s.name,
        utilization: s.utilization_rate,
        avgOccupancy: s.avg_occupancy,
        capacity: s.capacity
      }));
    
    res.json({ overview });
  } catch (error) {
    res.status(500).json({ error: 'Failed to load overview statistics' });
  }
});

module.exports = router; 