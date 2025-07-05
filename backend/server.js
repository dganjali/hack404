const express = require('express');
const { exec } = require('child_process');
const fs = require('fs');
const cors = require('cors');

const app = express();
const PORT = 3001;

// Cache for predictions
let predictionsCache = null;
let lastCacheTime = 0;
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

app.use(cors());

// Root route
app.get('/', (req, res) => {
  res.json({ message: 'Server is running! Use /api/predictions to get shelter predictions.' });
});

app.get('/api/predictions', (req, res) => {
  const now = Date.now();
  
  // Check if we have valid cached data
  if (predictionsCache && (now - lastCacheTime) < CACHE_DURATION) {
    console.log('Serving cached predictions');
    return res.json(predictionsCache);
  }

  console.log('Generating fresh predictions...');
  exec('python3 ../model/predict.py', { cwd: __dirname }, (error, stdout, stderr) => {
    if (error) {
      console.error('Python script error:', error);
      console.error('Stderr:', stderr);
      return res.status(500).send("Python script execution failed");
    }

    fs.readFile('../data/predictions.json', 'utf8', (err, data) => {
      if (err) {
        console.error('Failed to read predictions.json:', err);
        return res.status(500).send("Failed to read predictions");
      }

      try {
        const json = JSON.parse(data);
        // Cache the results
        predictionsCache = json;
        lastCacheTime = now;
        res.json(json);
      } catch (parseError) {
        console.error('JSON parse error:', parseError);
        res.status(500).send("Invalid JSON format in predictions file");
      }
    });
  });
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
