import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { ArrowLeft, Settings, User, Clock, DollarSign, FileText } from 'lucide-react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import ProcessFlow from '../components/ProcessFlow';
import StepDetails from '../components/StepDetails';

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

interface UserContext {
  province: string;
  city: string;
  business_type: string;
  industry: string;
  is_incorporated: boolean;
}

const ProcessViewer: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [process, setProcess] = useState<any>(null);
  const [personalizedProcess, setPersonalizedProcess] = useState<any>(null);
  const [userContext, setUserContext] = useState<UserContext>({
    province: 'Ontario',
    city: 'Toronto',
    business_type: 'sole_proprietor',
    industry: 'food',
    is_incorporated: false,
  });
  const [selectedStep, setSelectedStep] = useState<ProcessStep | null>(null);
  const [showPersonalization, setShowPersonalization] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProcess = async () => {
      try {
        const response = await axios.get(`/api/processes/${id}`);
        setProcess(response.data);
      } catch (error) {
        console.error('Error fetching process:', error);
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      fetchProcess();
    }
  }, [id]);

  useEffect(() => {
    const personalizeProcess = async () => {
      if (!process) return;

      try {
        const response = await axios.post('/api/processes/personalize', {
          process_id: id,
          user_context: userContext,
        });
        setPersonalizedProcess(response.data);
      } catch (error) {
        console.error('Error personalizing process:', error);
      }
    };

    if (process) {
      personalizeProcess();
    }
  }, [process, userContext, id]);

  const handleNodeClick = (node: any) => {
    const step = node.data.step;
    setSelectedStep(step);
  };

  const handleCloseDetails = () => {
    setSelectedStep(null);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading process...</p>
        </div>
      </div>
    );
  }

  if (!process) {
    return (
      <div className="text-center py-16">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Process Not Found</h2>
        <p className="text-gray-600 mb-8">The process you're looking for doesn't exist.</p>
        <Link to="/" className="btn-primary">
          Go Home
        </Link>
      </div>
    );
  }

  const steps = personalizedProcess?.filtered_steps || process.steps || [];
  const estimates = personalizedProcess || {};

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link to="/" className="text-gray-600 hover:text-gray-900">
                <ArrowLeft className="h-6 w-6" />
              </Link>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">{process.title}</h1>
                <p className="text-gray-600">{process.description}</p>
              </div>
            </div>
            <button
              onClick={() => setShowPersonalization(!showPersonalization)}
              className="btn-secondary flex items-center space-x-2"
            >
              <Settings className="h-4 w-4" />
              <span>Personalize</span>
            </button>
          </div>
        </div>
      </div>

      {/* Personalization Panel */}
      {showPersonalization && (
        <div className="bg-white border-b border-gray-200 p-6">
          <div className="container mx-auto">
            <h3 className="text-lg font-semibold mb-4">Personalize Your Process</h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Province
                </label>
                <select
                  value={userContext.province}
                  onChange={(e) => setUserContext({ ...userContext, province: e.target.value })}
                  className="input-field"
                >
                  <option value="Ontario">Ontario</option>
                  <option value="Quebec">Quebec</option>
                  <option value="British Columbia">British Columbia</option>
                  <option value="Alberta">Alberta</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  City
                </label>
                <input
                  type="text"
                  value={userContext.city}
                  onChange={(e) => setUserContext({ ...userContext, city: e.target.value })}
                  className="input-field"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Business Type
                </label>
                <select
                  value={userContext.business_type}
                  onChange={(e) => setUserContext({ ...userContext, business_type: e.target.value })}
                  className="input-field"
                >
                  <option value="sole_proprietor">Sole Proprietor</option>
                  <option value="partnership">Partnership</option>
                  <option value="corporation">Corporation</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Industry
                </label>
                <select
                  value={userContext.industry}
                  onChange={(e) => setUserContext({ ...userContext, industry: e.target.value })}
                  className="input-field"
                >
                  <option value="food">Food & Beverage</option>
                  <option value="retail">Retail</option>
                  <option value="services">Services</option>
                  <option value="technology">Technology</option>
                </select>
              </div>
              <div className="flex items-center">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={userContext.is_incorporated}
                    onChange={(e) => setUserContext({ ...userContext, is_incorporated: e.target.checked })}
                    className="mr-2"
                  />
                  <span className="text-sm font-medium text-gray-700">Incorporated</span>
                </label>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Estimates */}
      {estimates.estimated_time || estimates.estimated_cost ? (
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="container mx-auto">
            <div className="flex items-center justify-center space-x-8">
              {estimates.estimated_time && (
                <div className="flex items-center space-x-2">
                  <Clock className="h-5 w-5 text-blue-600" />
                  <span className="text-sm font-medium">Time: {estimates.estimated_time}</span>
                </div>
              )}
              {estimates.estimated_cost && (
                <div className="flex items-center space-x-2">
                  <DollarSign className="h-5 w-5 text-green-600" />
                  <span className="text-sm font-medium">Cost: {estimates.estimated_cost}</span>
                </div>
              )}
              {estimates.required_documents && estimates.required_documents.length > 0 && (
                <div className="flex items-center space-x-2">
                  <FileText className="h-5 w-5 text-purple-600" />
                  <span className="text-sm font-medium">
                    {estimates.required_documents.length} documents required
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}

      {/* Process Flow */}
      <div className="container mx-auto px-4 py-8">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold mb-4">Process Flow</h2>
          <ProcessFlow
            steps={steps}
            onNodeClick={handleNodeClick}
            selectedNode={selectedStep?.id || null}
          />
        </div>
      </div>

      {/* Step Details Sidebar */}
      {selectedStep && (
        <StepDetails step={selectedStep} onClose={handleCloseDetails} />
      )}
    </div>
  );
};

export default ProcessViewer; 