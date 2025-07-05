import express from 'express';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import auth from '../middleware/auth.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = express.Router();

// Load shelters data
async function loadShelters() {
  try {
    const data = await fs.readFile(path.join(__dirname, '../real_shelters.json'), 'utf8');
    return JSON.parse(data);
  } catch (err) {
    try {
      const data = await fs.readFile(path.join(__dirname, '../shelters.json'), 'utf8');
      return JSON.parse(data);
    } catch (err2) {
      return [];
    }
  }
}

// Load intake history
async function loadIntakeHistory() {
  try {
    const data = await fs.readFile(path.join(__dirname, '../real_intake_history.json'), 'utf8');
    return JSON.parse(data);
  } catch (err) {
    try {
      const data = await fs.readFile(path.join(__dirname, '../intake_history.json'), 'utf8');
      return JSON.parse(data);
    } catch (err2) {
      return [];
    }
  }
}

// Get shelters
router.get('/shelters', auth, async (req, res) => {
  try {
    const shelters = await loadShelters();
    res.json({ shelters });
  } catch (err) {
    res.status(500).json({ error: 'Failed to load shelters' });
  }
});

// Get intake history
router.get('/intake-history', auth, async (req, res) => {
  try {
    const history = await loadIntakeHistory();
    res.json({ history });
  } catch (err) {
    res.status(500).json({ error: 'Failed to load intake history' });
  }
});

// Get dashboard data
router.get('/dashboard-data', auth, async (req, res) => {
  try {
    const shelters = await loadShelters();
    const history = await loadIntakeHistory();
    
    res.json({
      shelters,
      history: history.slice(-30), // Last 30 records
      data_source: 'Real Toronto Data'
    });
  } catch (err) {
    res.status(500).json({ error: 'Failed to load dashboard data' });
  }
});

export default router; 