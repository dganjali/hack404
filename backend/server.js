import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import fetch from 'node-fetch';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

const app = express();
const PORT = process.env.PORT || 8000;
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/needsmatcher';
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8001';

app.use(cors());
app.use(express.json());

// Connect to MongoDB
mongoose.connect(MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB error:', err));

// Auth routes
app.use('/api/auth', (await import('./routes/auth.js')).default);

// Data routes (protected)
app.use('/api', (await import('./routes/data.js')).default);

// Proxy /api/forecast to ML microservice (protected)
const auth = (await import('./middleware/auth.js')).default;
app.post('/api/forecast', auth, async (req, res) => {
  try {
    const mlRes = await fetch(`${ML_SERVICE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: req.body.features })
    });
    const data = await mlRes.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: 'Prediction failed' });
  }
});

// Serve static files
app.use(express.static(path.join(__dirname, '../public')));

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Node.js backend is running!' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 