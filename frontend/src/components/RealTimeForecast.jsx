import React, { useEffect, useState } from 'react';

const RealTimeForecast = ({ token }) => {
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchForecast = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/real-time-forecast`, {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        setForecast(data.current_prediction);
      } catch (err) {
        setError(err.message || 'Failed to fetch forecast');
      } finally {
        setLoading(false);
      }
    };
    if (token) fetchForecast();
  }, [token]);

  if (!token) return <div className="p-4 text-red-500">Please log in to see real-time forecast.</div>;
  if (loading) return <div className="p-4">Loading real-time forecast...</div>;
  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!forecast) return null;

  return (
    <div className="bg-white rounded shadow p-6 mb-6">
      <h2 className="text-xl font-bold mb-2">Real-Time Shelter Demand Forecast</h2>
      <div className="mb-2 text-gray-600 text-sm">{forecast.timestamp && `As of ${new Date(forecast.timestamp).toLocaleString()}`}</div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="font-semibold">Beds Needed</div>
          <div className="text-2xl">{forecast.beds_needed}</div>
        </div>
        <div>
          <div className="font-semibold">Meals Needed</div>
          <div className="text-2xl">{forecast.meals_needed}</div>
        </div>
        <div>
          <div className="font-semibold">Kits Needed</div>
          <div className="text-2xl">{forecast.kits_needed}</div>
        </div>
        <div>
          <div className="font-semibold">Total Occupancy</div>
          <div className="text-2xl">{forecast.total_occupancy}</div>
        </div>
      </div>
      <div className="mt-4 text-sm text-gray-500">
        Confidence: <span className="font-semibold">{Math.round(forecast.confidence * 100)}%</span><br />
        Data Source: {forecast.data_source}
      </div>
    </div>
  );
};

export default RealTimeForecast; 