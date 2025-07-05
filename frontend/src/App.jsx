import React, { useState, useEffect } from 'react';
import Login from './components/Login';
import Register from './components/Register';
import Header from './components/Header';
import ShelterCard from './components/ShelterCard';
import ForecastChart from './components/ForecastChart';
import TransferTable from './components/TransferTable';
import RealTimeForecast from './components/RealTimeForecast';
import API_BASE_URL from './config';
import { RefreshCw, TrendingUp, Package, Users, MapPin, Database } from 'lucide-react';

function App() {
  const [user, setUser] = useState(null);
  const [showRegister, setShowRegister] = useState(false);
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dataSource, setDataSource] = useState('Mock Data');

  // Check for existing token on app load
  useEffect(() => {
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    
    if (token && userData) {
      setUser(JSON.parse(userData));
      fetchDashboardData(token);
    }
  }, []);

  const fetchDashboardData = async (token) => {
    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/dashboard-data`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setDashboardData(data);
        setDataSource(data.data_source || 'Mock Data');
      } else if (response.status === 401) {
        // Token expired or invalid
        handleLogout();
      } else {
        setError('Failed to load dashboard data');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = (userData) => {
    setUser(userData);
    fetchDashboardData(localStorage.getItem('token'));
  };

  const handleRegister = (message) => {
    alert(message);
    setShowRegister(false);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setDashboardData(null);
  };

  const handleSwitchToRegister = () => {
    setShowRegister(true);
  };

  const handleSwitchToLogin = () => {
    setShowRegister(false);
  };

  // Show authentication screens if not logged in
  if (!user) {
    return showRegister ? (
      <Register onRegister={handleRegister} onSwitchToLogin={handleSwitchToLogin} />
    ) : (
      <Login onLogin={handleLogin} onSwitchToRegister={handleSwitchToRegister} />
    );
  }

  // Show dashboard if logged in
  return (
    <div className="min-h-screen bg-gray-100">
      <Header user={user} onLogout={handleLogout} />
      
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {loading && (
          <div className="text-center py-4">
            <div className="inline-flex items-center px-4 py-2 font-semibold leading-6 text-sm shadow rounded-md text-white bg-indigo-500 hover:bg-indigo-400 transition ease-in-out duration-150 cursor-not-allowed">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Loading dashboard...
            </div>
          </div>
        )}

        {error && (
          <div className="mb-4 bg-red-50 border border-red-200 rounded-md p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">
                  Error loading dashboard
                </h3>
                <div className="mt-2 text-sm text-red-700">
                  {error}
                </div>
              </div>
            </div>
          </div>
        )}

        {dashboardData && (
          <div className="space-y-6">
            {/* Real-Time Forecast */}
            <RealTimeForecast token={localStorage.getItem('token')} />
            {/* Data Source Info */}
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                      Data Source
                    </h3>
                    <p className="mt-1 text-sm text-gray-500">
                      {dataSource}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-semibold text-gray-900">
                      {dashboardData.shelters?.length || 0}
                    </p>
                    <p className="text-sm text-gray-500">Active Shelters</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Shelter Overview */}
            <div className="bg-white shadow overflow-hidden sm:rounded-md">
              <div className="px-4 py-5 sm:px-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900">
                  Shelter Overview
                </h3>
                <p className="mt-1 max-w-2xl text-sm text-gray-500">
                  Current inventory and capacity across all shelters
                </p>
              </div>
              <div className="border-t border-gray-200">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
                  {dashboardData.shelters?.map((shelter) => (
                    <ShelterCard key={shelter.id} shelter={shelter} />
                  ))}
                </div>
              </div>
            </div>

            {/* Forecast Chart */}
            <div className="bg-white shadow overflow-hidden sm:rounded-lg">
              <div className="px-4 py-5 sm:px-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900">
                  Demand Forecast
                </h3>
                <p className="mt-1 max-w-2xl text-sm text-gray-500">
                  Predicted resource needs for tomorrow
                </p>
              </div>
              <div className="border-t border-gray-200">
                <ForecastChart forecasts={dashboardData.forecasts} shelters={dashboardData.shelters} />
              </div>
            </div>

            {/* Transfer Recommendations */}
            <div className="bg-white shadow overflow-hidden sm:rounded-lg">
              <div className="px-4 py-5 sm:px-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900">
                  Transfer Recommendations
                </h3>
                <p className="mt-1 max-w-2xl text-sm text-gray-500">
                  Optimized resource transfers to reduce shortages
                </p>
              </div>
              <div className="border-t border-gray-200">
                <TransferTable transfers={dashboardData.transfers} shortagesReduced={dashboardData.shortages_reduced} />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App; 