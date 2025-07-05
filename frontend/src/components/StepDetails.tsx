import React from 'react';
import { FileText, Clock, DollarSign, AlertCircle, CheckCircle } from 'lucide-react';

interface ProcessStep {
  id: string;
  type: string;
  title: string;
  description?: string;
  cost?: string;
  duration?: string;
  required_documents?: string[];
  conditions?: string[];
  depends_on?: string[];
  outputs?: string[];
  url?: string;
}

interface StepDetailsProps {
  step: ProcessStep | null;
  onClose: () => void;
}

const StepDetails: React.FC<StepDetailsProps> = ({ step, onClose }) => {
  if (!step) {
    return null;
  }

  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'action':
        return <CheckCircle className="h-5 w-5 text-blue-600" />;
      case 'document':
        return <FileText className="h-5 w-5 text-green-600" />;
      case 'fee':
        return <DollarSign className="h-5 w-5 text-yellow-600" />;
      case 'wait':
        return <Clock className="h-5 w-5 text-purple-600" />;
      case 'decision':
        return <AlertCircle className="h-5 w-5 text-orange-600" />;
      default:
        return <FileText className="h-5 w-5 text-gray-600" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'action':
        return 'bg-blue-50 border-blue-200 text-blue-800';
      case 'document':
        return 'bg-green-50 border-green-200 text-green-800';
      case 'fee':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'wait':
        return 'bg-purple-50 border-purple-200 text-purple-800';
      case 'decision':
        return 'bg-orange-50 border-orange-200 text-orange-800';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  return (
    <div className="fixed inset-y-0 right-0 w-96 bg-white shadow-lg border-l border-gray-200 overflow-y-auto">
      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            {getTypeIcon(step.type)}
            <div>
              <h2 className="text-lg font-semibold text-gray-900">{step.title}</h2>
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getTypeColor(step.type)}`}>
                {step.type}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Description */}
        {step.description && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-900 mb-2">Description</h3>
            <p className="text-sm text-gray-600">{step.description}</p>
          </div>
        )}

        {/* Cost and Duration */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          {step.cost && (
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-1">Cost</h3>
              <p className="text-sm text-gray-600">{step.cost}</p>
            </div>
          )}
          {step.duration && (
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-1">Duration</h3>
              <p className="text-sm text-gray-600">{step.duration}</p>
            </div>
          )}
        </div>

        {/* Required Documents */}
        {step.required_documents && step.required_documents.length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-900 mb-2">Required Documents</h3>
            <ul className="space-y-1">
              {step.required_documents.map((doc, index) => (
                <li key={index} className="flex items-center space-x-2">
                  <FileText className="h-4 w-4 text-green-600" />
                  <span className="text-sm text-gray-600">{doc}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Conditions */}
        {step.conditions && step.conditions.length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-900 mb-2">Conditions</h3>
            <ul className="space-y-1">
              {step.conditions.map((condition, index) => (
                <li key={index} className="flex items-center space-x-2">
                  <AlertCircle className="h-4 w-4 text-orange-600" />
                  <span className="text-sm text-gray-600">{condition}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Dependencies */}
        {step.depends_on && step.depends_on.length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-900 mb-2">Dependencies</h3>
            <ul className="space-y-1">
              {step.depends_on.map((dep, index) => (
                <li key={index} className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-blue-600" />
                  <span className="text-sm text-gray-600">Step {dep}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Outputs */}
        {step.outputs && step.outputs.length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-900 mb-2">Outputs</h3>
            <ul className="space-y-1">
              {step.outputs.map((output, index) => (
                <li key={index} className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                  <span className="text-sm text-gray-600">{output}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* External Link */}
        {step.url && (
          <div className="mt-6 pt-6 border-t border-gray-200">
            <a
              href={step.url}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-primary w-full text-center"
            >
              View Official Page
            </a>
          </div>
        )}
      </div>
    </div>
  );
};

export default StepDetails; 