import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import ChartsPage from './components/ChartsPage';
import PredictionForm from './components/PredictionForm';
import './index.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [players, setPlayers] = useState([]);
  const [teams, setTeams] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [playersResponse, teamsResponse] = await Promise.all([
        axios.get(`${API_BASE_URL}/players`),
        axios.get(`${API_BASE_URL}/teams`)
      ]);
      
      setPlayers(playersResponse.data);
      setTeams(teamsResponse.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch data. Make sure the backend server is running on port 8000.');
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
    }
  };

  const renderCurrentPage = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full"
          />
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center p-8 bg-red-50 rounded-lg border border-red-200"
          >
            <div className="text-red-600 text-6xl mb-4">⚠️</div>
            <h2 className="text-xl font-semibold text-red-800 mb-2">Connection Error</h2>
            <p className="text-red-600 mb-4">{error}</p>
            <button
              onClick={fetchData}
              className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
            >
              Retry Connection
            </button>
          </motion.div>
        </div>
      );
    }

    switch (currentPage) {
      case 'dashboard':
        return <Dashboard players={players} teams={teams} />;
      case 'charts':
        return <ChartsPage players={players} teams={teams} />;
      case 'prediction':
        return <PredictionForm />;
      default:
        return <Dashboard players={players} teams={teams} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar currentPage={currentPage} onPageChange={setCurrentPage} />
      <main className="container mx-auto px-4 py-6">
        {renderCurrentPage()}
      </main>
    </div>
  );
}

export default App;