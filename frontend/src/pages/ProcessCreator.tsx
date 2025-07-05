import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Globe, Upload, FileText, Loader } from 'lucide-react';
import axios from 'axios';

interface ScrapingRequest {
  urls: string[];
  process_name: string;
  description?: string;
}

const ProcessCreator: React.FC = () => {
  const [scrapingRequest, setScrapingRequest] = useState<ScrapingRequest>({
    urls: [''],
    process_name: '',
    description: '',
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUrlChange = (index: number, value: string) => {
    const newUrls = [...scrapingRequest.urls];
    newUrls[index] = value;
    setScrapingRequest({ ...scrapingRequest, urls: newUrls });
  };

  const addUrl = () => {
    setScrapingRequest({
      ...scrapingRequest,
      urls: [...scrapingRequest.urls, ''],
    });
  };

  const removeUrl = (index: number) => {
    const newUrls = scrapingRequest.urls.filter((_, i) => i !== index);
    setScrapingRequest({ ...scrapingRequest, urls: newUrls });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      // Filter out empty URLs
      const validUrls = scrapingRequest.urls.filter(url => url.trim() !== '');
      
      if (validUrls.length === 0) {
        throw new Error('Please provide at least one URL');
      }

      if (!scrapingRequest.process_name.trim()) {
        throw new Error('Please provide a process name');
      }

      const response = await axios.post('/api/processes/scrape', {
        urls: validUrls,
        process_name: scrapingRequest.process_name,
        description: scrapingRequest.description,
      });

      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'An error occurred');
    } finally {
      setIsProcessing(false);
    }
  };

  const createSampleProcess = async () => {
    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('/api/processes/sample/food-truck');
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'An error occurred');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center space-x-4">
            <Link to="/" className="text-gray-600 hover:text-gray-900">
              <ArrowLeft className="h-6 w-6" />
            </Link>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Create New Process</h1>
              <p className="text-gray-600">Scrape government websites to create interactive process maps</p>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Quick Start */}
          <div className="card mb-8">
            <h2 className="text-xl font-semibold mb-4">Quick Start</h2>
            <p className="text-gray-600 mb-4">
              Create a sample food truck process to see how it works:
            </p>
            <button
              onClick={createSampleProcess}
              disabled={isProcessing}
              className="btn-primary flex items-center space-x-2"
            >
              {isProcessing ? (
                <Loader className="h-4 w-4 animate-spin" />
              ) : (
                <FileText className="h-4 w-4" />
              )}
              <span>Create Sample Process</span>
            </button>
          </div>

          {/* Scraping Form */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Scrape Government Websites</h2>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Process Name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Process Name *
                </label>
                <input
                  type="text"
                  value={scrapingRequest.process_name}
                  onChange={(e) => setScrapingRequest({ ...scrapingRequest, process_name: e.target.value })}
                  placeholder="e.g., Starting a Food Truck Business"
                  className="input-field"
                  required
                />
              </div>

              {/* Description */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Description
                </label>
                <textarea
                  value={scrapingRequest.description}
                  onChange={(e) => setScrapingRequest({ ...scrapingRequest, description: e.target.value })}
                  placeholder="Brief description of the process..."
                  className="input-field"
                  rows={3}
                />
              </div>

              {/* URLs */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Government Website URLs *
                </label>
                <div className="space-y-3">
                  {scrapingRequest.urls.map((url, index) => (
                    <div key={index} className="flex space-x-2">
                      <input
                        type="url"
                        value={url}
                        onChange={(e) => handleUrlChange(index, e.target.value)}
                        placeholder="https://www.ontario.ca/page/start-business"
                        className="input-field flex-1"
                        required
                      />
                      {scrapingRequest.urls.length > 1 && (
                        <button
                          type="button"
                          onClick={() => removeUrl(index)}
                          className="px-3 py-2 text-red-600 hover:text-red-700"
                        >
                          Remove
                        </button>
                      )}
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={addUrl}
                    className="text-primary-600 hover:text-primary-700 text-sm font-medium"
                  >
                    + Add another URL
                  </button>
                </div>
                <p className="text-sm text-gray-500 mt-2">
                  Add URLs from government websites that contain information about your process.
                </p>
              </div>

              {/* Submit */}
              <div className="flex items-center space-x-4">
                <button
                  type="submit"
                  disabled={isProcessing}
                  className="btn-primary flex items-center space-x-2"
                >
                  {isProcessing ? (
                    <Loader className="h-4 w-4 animate-spin" />
                  ) : (
                    <Globe className="h-4 w-4" />
                  )}
                  <span>{isProcessing ? 'Processing...' : 'Create Process'}</span>
                </button>
                <Link to="/" className="btn-secondary">
                  Cancel
                </Link>
              </div>
            </form>
          </div>

          {/* Error */}
          {error && (
            <div className="card border-red-200 bg-red-50">
              <h3 className="text-lg font-semibold text-red-800 mb-2">Error</h3>
              <p className="text-red-700">{error}</p>
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="card border-green-200 bg-green-50">
              <h3 className="text-lg font-semibold text-green-800 mb-2">Process Created Successfully!</h3>
              <p className="text-green-700 mb-4">
                Your process has been created with {result.steps?.length || 0} steps.
              </p>
              <div className="flex space-x-4">
                <Link
                  to={`/process/${result.process_id}`}
                  className="btn-primary"
                >
                  View Process
                </Link>
                <button
                  onClick={() => {
                    setResult(null);
                    setScrapingRequest({ urls: [''], process_name: '', description: '' });
                  }}
                  className="btn-secondary"
                >
                  Create Another
                </button>
              </div>
            </div>
          )}

          {/* Tips */}
          <div className="card bg-blue-50 border-blue-200">
            <h3 className="text-lg font-semibold text-blue-800 mb-2">Tips for Better Results</h3>
            <ul className="text-blue-700 space-y-1 text-sm">
              <li>• Use official government websites (.gov, .gc.ca, .gov.on.ca, etc.)</li>
              <li>• Include multiple pages from the same process (forms, guides, FAQs)</li>
              <li>• Provide a clear, descriptive process name</li>
              <li>• The AI will automatically extract and classify steps from the content</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProcessCreator; 