import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  LineElement,
  PointElement,
} from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  LineElement,
  PointElement
);

const ForecastChart = ({ history, forecasts }) => {
  // Process historical data
  const processedHistory = history.reduce((acc, record) => {
    const date = record.date;
    if (!acc[date]) {
      acc[date] = { beds: 0, meals: 0, kits: 0 };
    }
    acc[date].beds += record.beds_needed;
    acc[date].meals += record.meals_needed;
    acc[date].kits += record.kits_needed;
    return acc;
  }, {});

  const dates = Object.keys(processedHistory).sort();
  const lastDate = dates[dates.length - 1];

  // Calculate tomorrow's forecast totals
  const tomorrowForecast = Object.values(forecasts).reduce(
    (acc, forecast) => {
      acc.beds += forecast.predicted_beds_needed;
      acc.meals += forecast.predicted_meals_needed;
      acc.kits += forecast.predicted_kits_needed;
      return acc;
    },
    { beds: 0, meals: 0, kits: 0 }
  );

  // Create chart data
  const chartData = {
    labels: [...dates, 'Tomorrow'],
    datasets: [
      {
        label: 'Beds Needed',
        data: [
          ...dates.map(date => processedHistory[date].beds),
          tomorrowForecast.beds
        ],
        backgroundColor: 'rgba(59, 130, 246, 0.8)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 1,
      },
      {
        label: 'Meals Needed',
        data: [
          ...dates.map(date => processedHistory[date].meals),
          tomorrowForecast.meals
        ],
        backgroundColor: 'rgba(34, 197, 94, 0.8)',
        borderColor: 'rgba(34, 197, 94, 1)',
        borderWidth: 1,
      },
      {
        label: 'Kits Needed',
        data: [
          ...dates.map(date => processedHistory[date].kits),
          tomorrowForecast.kits
        ],
        backgroundColor: 'rgba(168, 85, 247, 0.8)',
        borderColor: 'rgba(168, 85, 247, 1)',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Daily Resource Demand (Last 7 Days + Tomorrow)',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Quantity Needed',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Date',
        },
      },
    },
  };

  // Create trend line data
  const trendData = {
    labels: dates,
    datasets: [
      {
        label: 'Beds Trend',
        data: dates.map(date => processedHistory[date].beds),
        borderColor: 'rgba(59, 130, 246, 1)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: false,
      },
      {
        label: 'Meals Trend',
        data: dates.map(date => processedHistory[date].meals),
        borderColor: 'rgba(34, 197, 94, 1)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
        fill: false,
      },
      {
        label: 'Kits Trend',
        data: dates.map(date => processedHistory[date].kits),
        borderColor: 'rgba(168, 85, 247, 1)',
        backgroundColor: 'rgba(168, 85, 247, 0.1)',
        tension: 0.4,
        fill: false,
      },
    ],
  };

  const trendOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Historical Trend (Last 7 Days)',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Quantity Needed',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Date',
        },
      },
    },
  };

  return (
    <div className="space-y-6">
      {/* Bar Chart */}
      <div className="h-80">
        <Bar data={chartData} options={options} />
      </div>
      
      {/* Line Chart */}
      <div className="h-80">
        <Line data={trendData} options={trendOptions} />
      </div>
      
      {/* Forecast Summary */}
      <div className="grid grid-cols-3 gap-4 mt-6">
        <div className="bg-blue-50 rounded-lg p-4">
          <h4 className="font-semibold text-blue-900 mb-2">Tomorrow's Bed Forecast</h4>
          <p className="text-2xl font-bold text-blue-600">{tomorrowForecast.beds}</p>
          <p className="text-sm text-blue-600">Total beds needed across all shelters</p>
        </div>
        
        <div className="bg-green-50 rounded-lg p-4">
          <h4 className="font-semibold text-green-900 mb-2">Tomorrow's Meal Forecast</h4>
          <p className="text-2xl font-bold text-green-600">{tomorrowForecast.meals}</p>
          <p className="text-sm text-green-600">Total meals needed across all shelters</p>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-4">
          <h4 className="font-semibold text-purple-900 mb-2">Tomorrow's Kit Forecast</h4>
          <p className="text-2xl font-bold text-purple-600">{tomorrowForecast.kits}</p>
          <p className="text-sm text-purple-600">Total kits needed across all shelters</p>
        </div>
      </div>
    </div>
  );
};

export default ForecastChart; 