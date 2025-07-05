import React from 'react';
import { ArrowRight, Package, Truck } from 'lucide-react';

const TransferTable = ({ transfers, shelters }) => {
  const getShelterName = (shelterId) => {
    const shelter = shelters.find(s => s.id === shelterId);
    return shelter ? shelter.name : shelterId;
  };

  const getItemIcon = (item) => {
    switch (item) {
      case 'beds':
        return '🛏️';
      case 'meals':
        return '🍽️';
      case 'kits':
        return '📦';
      default:
        return '📦';
    }
  };

  const getItemColor = (item) => {
    switch (item) {
      case 'beds':
        return 'bg-blue-100 text-blue-800';
      case 'meals':
        return 'bg-green-100 text-green-800';
      case 'kits':
        return 'bg-purple-100 text-purple-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  if (transfers.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="bg-green-100 rounded-full p-4 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
          <Truck className="text-green-600 w-8 h-8" />
        </div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">No Transfers Needed</h3>
        <p className="text-gray-600">All shelters have sufficient resources for tomorrow's predicted demand.</p>
      </div>
    );
  }

  return (
    <div>
      {/* Transfer Summary */}
      <div className="mb-6 p-4 bg-blue-50 rounded-lg">
        <div className="flex items-center">
          <Truck className="text-blue-600 w-5 h-5 mr-2" />
          <h3 className="font-semibold text-blue-900">Transfer Summary</h3>
        </div>
        <p className="text-sm text-blue-700 mt-1">
          {transfers.length} transfer{transfers.length !== 1 ? 's' : ''} planned to optimize resource allocation
        </p>
      </div>

      {/* Transfer Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                From Shelter
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                To Shelter
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Resource
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Amount
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {transfers.map((transfer, index) => (
              <tr key={index} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">
                    {getShelterName(transfer.from)}
                  </div>
                  <div className="text-sm text-gray-500">
                    {transfer.from}
                  </div>
                </td>
                
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <ArrowRight className="w-4 h-4 text-gray-400 mr-2" />
                    <div>
                      <div className="text-sm font-medium text-gray-900">
                        {getShelterName(transfer.to)}
                      </div>
                      <div className="text-sm text-gray-500">
                        {transfer.to}
                      </div>
                    </div>
                  </div>
                </td>
                
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <span className="mr-2">{getItemIcon(transfer.item)}</span>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getItemColor(transfer.item)}`}>
                      {transfer.item.charAt(0).toUpperCase() + transfer.item.slice(1)}
                    </span>
                  </div>
                </td>
                
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-semibold text-gray-900">
                    {transfer.amount}
                  </div>
                  <div className="text-sm text-gray-500">
                    units
                  </div>
                </td>
                
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                    Pending
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Transfer Actions */}
      <div className="mt-6 flex justify-end space-x-3">
        <button className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500">
          Export Plan
        </button>
        <button className="px-4 py-2 bg-primary-600 border border-transparent rounded-md text-sm font-medium text-white hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500">
          Execute Transfers
        </button>
      </div>

      {/* Transfer Statistics */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg p-4 border border-gray-200">
          <div className="flex items-center">
            <Package className="text-blue-600 w-5 h-5 mr-2" />
            <div>
              <p className="text-sm font-medium text-gray-600">Total Items</p>
              <p className="text-lg font-semibold text-gray-900">
                {transfers.reduce((sum, t) => sum + t.amount, 0)}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg p-4 border border-gray-200">
          <div className="flex items-center">
            <Truck className="text-green-600 w-5 h-5 mr-2" />
            <div>
              <p className="text-sm font-medium text-gray-600">Unique Shelters</p>
              <p className="text-lg font-semibold text-gray-900">
                {new Set([...transfers.map(t => t.from), ...transfers.map(t => t.to)]).size}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg p-4 border border-gray-200">
          <div className="flex items-center">
            <ArrowRight className="text-purple-600 w-5 h-5 mr-2" />
            <div>
              <p className="text-sm font-medium text-gray-600">Resource Types</p>
              <p className="text-lg font-semibold text-gray-900">
                {new Set(transfers.map(t => t.item)).size}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TransferTable; 