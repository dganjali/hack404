import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Zap, Globe, FileText, TrendingUp, ArrowRight, Play } from 'lucide-react';
import axios from 'axios';

interface Process {
  id: string;
  title: string;
  description: string;
  steps_count: number;
}

const Home: React.FC = () => {
  const [processes, setProcesses] = useState<Process[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProcesses = async () => {
      try {
        const response = await axios.get('/api/processes/');
        setProcesses(response.data.processes || []);
      } catch (error) {
        console.error('Error fetching processes:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchProcesses();
  }, []);

  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="text-center py-16">
        <div className="max-w-4xl mx-auto">
          <div className="flex justify-center mb-6">
            <div className="bg-primary-100 p-3 rounded-full">
              <Zap className="h-8 w-8 text-primary-600" />
            </div>
          </div>
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            ProcessUnravel Pro
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            Transform complex government processes into interactive, personalized decision trees. 
            Navigate bureaucracy with AI-powered clarity.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/create" className="btn-primary text-lg px-8 py-3">
              Create Your Process
            </Link>
            <button className="btn-secondary text-lg px-8 py-3 flex items-center justify-center">
              <Play className="h-5 w-5 mr-2" />
              Watch Demo
            </button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-white">
        <div className="max-w-6xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
            How It Works
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-primary-100 p-4 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <Globe className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Scrape & Parse</h3>
              <p className="text-gray-600">
                AI agents automatically extract and classify steps from government websites, 
                PDFs, and official documents.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-success-100 p-4 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <FileText className="h-8 w-8 text-success-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Build Decision Trees</h3>
              <p className="text-gray-600">
                Transform static information into interactive flowcharts with dependencies, 
                conditions, and personalized paths.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-warning-100 p-4 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <TrendingUp className="h-8 w-8 text-warning-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Navigate & Execute</h3>
              <p className="text-gray-600">
                Get personalized guidance with time estimates, costs, and required documents 
                for your specific situation.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Sample Process Section */}
      <section className="py-16">
        <div className="max-w-6xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
            Sample Processes
          </h2>
          
          {loading ? (
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
              <p className="mt-4 text-gray-600">Loading processes...</p>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Food Truck Process */}
              <div className="card hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Food Truck Business</h3>
                  <span className="bg-success-100 text-success-800 text-xs px-2 py-1 rounded-full">
                    Sample
                  </span>
                </div>
                <p className="text-gray-600 mb-4">
                  Complete process for starting a food truck business in Toronto, including 
                  licensing, health inspections, and permits.
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">8 steps</span>
                  <Link 
                    to="/process/food-truck-sample" 
                    className="flex items-center text-primary-600 hover:text-primary-700"
                  >
                    View Process
                    <ArrowRight className="h-4 w-4 ml-1" />
                  </Link>
                </div>
              </div>

              {/* Add more sample processes here */}
              <div className="card hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Business Registration</h3>
                  <span className="bg-primary-100 text-primary-800 text-xs px-2 py-1 rounded-full">
                    Coming Soon
                  </span>
                </div>
                <p className="text-gray-600 mb-4">
                  Register your business with the province and get your business number 
                  from the Canada Revenue Agency.
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">5 steps</span>
                  <span className="text-sm text-gray-400">Coming soon</span>
                </div>
              </div>

              <div className="card hover:shadow-lg transition-shadow">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Building Permits</h3>
                  <span className="bg-primary-100 text-primary-800 text-xs px-2 py-1 rounded-full">
                    Coming Soon
                  </span>
                </div>
                <p className="text-gray-600 mb-4">
                  Navigate the building permit process for renovations, additions, 
                  or new construction projects.
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">12 steps</span>
                  <span className="text-sm text-gray-400">Coming soon</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-primary-600 text-white">
        <div className="max-w-4xl mx-auto text-center px-4">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Simplify Your Process?
          </h2>
          <p className="text-xl mb-8 opacity-90">
            Create your first personalized process map in minutes.
          </p>
          <Link to="/create" className="bg-white text-primary-600 font-semibold py-3 px-8 rounded-lg hover:bg-gray-100 transition-colors">
            Get Started Now
          </Link>
        </div>
      </section>
    </div>
  );
};

export default Home; 