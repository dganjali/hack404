import React from 'react';
import { Bed, Utensils, Package, TrendingUp, TrendingDown } from 'lucide-react';

const ShelterCard = ({ shelter, forecast }) => {
  if (!forecast) {
    return (
      <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
        <h3 className="font-semibold text-gray-900 mb-2">{shelter.name}</h3>
        <p className="text-sm text-gray-600">No forecast data available</p>
      </div>
    );
  }

  const calculateShortage = (current, predicted) => {
    return Math.max(0, predicted - current);
  };

  const calculateSurplus = (current, predicted) => {
    return Math.max(0, current - predicted);
  };

  const getStatusColor = (shortage) => {
    if (shortage === 0) return 'text-green-600';
    if (shortage <= 5) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getStatusIcon = (shortage) => {
    if (shortage === 0) return <TrendingUp className="w-4 h-4" />;
    return <TrendingDown className="w-4 h-4" />;
  };

  const bedsShortage = calculateShortage(shelter.current_beds, forecast.predicted_beds_needed);
  const mealsShortage = calculateShortage(shelter.current_meals, forecast.predicted_meals_needed);
  const kitsShortage = calculateShortage(shelter.current_kits, forecast.predicted_kits_needed);

  const totalShortage = bedsShortage + mealsShortage + kitsShortage;

  return (
    <div className="bg-white rounded-lg p-4 border border-gray-200 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-900 text-sm">{shelter.name}</h3>
        <div className={`flex items-center ${getStatusColor(totalShortage)}`}>
          {getStatusIcon(totalShortage)}
        </div>
      </div>

      <div className="space-y-3">
        {/* Beds */}
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Bed className="w-4 h-4 text-blue-600 mr-2" />
            <span className="text-sm text-gray-600">Beds</span>
          </div>
          <div className="text-right">
            <div className="text-sm font-medium text-gray-900">
              {shelter.current_beds} / {forecast.predicted_beds_needed}
            </div>
            {bedsShortage > 0 && (
              <div className="text-xs text-red-600">-{bedsShortage}</div>
            )}
          </div>
        </div>

        {/* Meals */}
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Utensils className="w-4 h-4 text-green-600 mr-2" />
            <span className="text-sm text-gray-600">Meals</span>
          </div>
          <div className="text-right">
            <div className="text-sm font-medium text-gray-900">
              {shelter.current_meals} / {forecast.predicted_meals_needed}
            </div>
            {mealsShortage > 0 && (
              <div className="text-xs text-red-600">-{mealsShortage}</div>
            )}
          </div>
        </div>

        {/* Kits */}
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Package className="w-4 h-4 text-purple-600 mr-2" />
            <span className="text-sm text-gray-600">Kits</span>
          </div>
          <div className="text-right">
            <div className="text-sm font-medium text-gray-900">
              {shelter.current_kits} / {forecast.predicted_kits_needed}
            </div>
            {kitsShortage > 0 && (
              <div className="text-xs text-red-600">-{kitsShortage}</div>
            )}
          </div>
        </div>
      </div>

      {/* Capacity indicator */}
      <div className="mt-3 pt-3 border-t border-gray-100">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Capacity</span>
          <span>{shelter.current_beds}/{shelter.capacity}</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
          <div
            className="bg-blue-600 h-1.5 rounded-full"
            style={{ width: `${(shelter.current_beds / shelter.capacity) * 100}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
};

export default ShelterCard; 