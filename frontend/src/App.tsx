import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import ProcessViewer from './pages/ProcessViewer';
import ProcessCreator from './pages/ProcessCreator';
import './App.css';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/process/:id" element={<ProcessViewer />} />
            <Route path="/create" element={<ProcessCreator />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App; 