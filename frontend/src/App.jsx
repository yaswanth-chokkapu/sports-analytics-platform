import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

const API_BASE = ' http://127.0.0.1:8000';

// Icons
const TrophyIcon = () => (
  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M10 2L3 7v11a1 1 0 001 1h12a1 1 0 001-1V7l-7-5zM9 9a1 1 0 012 0v4a1 1 0 11-2 0V9z" clipRule="evenodd" />
  </svg>
);

const UserIcon = () => (
  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
  </svg>
);

const ChartIcon = () => (
  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
    <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
  </svg>
);

const BoltIcon = () => (
  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
  </svg>
);

const RefreshIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
  </svg>
);

const SearchIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
  </svg>
);

// Mock data for charts
const performanceData = [
  { name: 'Week 1', goals: 4, assists: 2, performance: 8.2 },
  { name: 'Week 2', goals: 2, assists: 5, performance: 7.8 },
  { name: 'Week 3', goals: 6, assists: 1, performance: 9.1 },
  { name: 'Week 4', goals: 3, assists: 4, performance: 8.5 },
  { name: 'Week 5', goals: 5, assists: 3, performance: 8.9 },
];

const teamChemistryData = [
  { name: 'Defense', value: 85 },
  { name: 'Midfield', value: 92 },
  { name: 'Attack', value: 78 },
];

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444'];

const SportsAnalyticsDashboard = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [players, setPlayers] = useState([]);
  const [teams, setTeams] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [wsMessages, setWsMessages] = useState([]);
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [predictionForm, setPredictionForm] = useState({
    player_id: '',
    goals: 0,
    assists: 0,
    pass_accuracy: 0,
    minutes_played: 0
  });
  const [searchTerm, setSearchTerm] = useState('');
  const wsRef = useRef(null);

  useEffect(() => {
    fetchData();
    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [playersRes, teamsRes, predictionsRes] = await Promise.all([
        fetch(`${API_BASE}/players`),
        fetch(`${API_BASE}/teams`),
        fetch(`${API_BASE}/predictions`, {
          headers: { Authorization: 'Bearer mock-token' }
        })
      ]);

      const playersData = await playersRes.json();
      const teamsData = await teamsRes.json();
      const predictionsData = await predictionsRes.json();

      setPlayers(playersData);
      setTeams(teamsData);
      setPredictions(predictionsData);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket(`ws://localhost:8000/ws/user_${Date.now()}`);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsStatus('connected');
        console.log('WebSocket connected');
      };

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        setWsMessages(prev => [...prev.slice(-9), message]);
        
        if (message.type === 'prediction') {
          setPredictions(prev => [message.data, ...prev]);
        }
      };

      ws.onclose = () => {
        setWsStatus('disconnected');
        console.log('WebSocket disconnected');
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsStatus('error');
      };
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      setWsStatus('error');
    }
  };

  const handlePrediction = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer mock-token'
        },
        body: JSON.stringify(predictionForm)
      });

      const prediction = await response.json();
      setPredictions(prev => [prediction, ...prev]);
      
      // Reset form
      setPredictionForm({
        player_id: '',
        goals: 0,
        assists: 0,
        pass_accuracy: 0,
        minutes_played: 0
      });
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredPlayers = players.filter(player =>
    player.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    player.team.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const StatCard = ({ title, value, subtitle, icon: Icon, color = "blue" }) => (
    <div className={`bg-white rounded-xl p-6 shadow-sm border border-gray-100 hover:shadow-md transition-shadow`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className={`text-2xl font-bold text-${color}-600`}>{value}</p>
          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 bg-${color}-100 rounded-lg`}>
          <Icon className={`text-${color}-600`} />
        </div>
      </div>
    </div>
  );

  const DashboardContent = () => (
    <div className="space-y-6">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Players"
          value={players.length}
          subtitle="Active players"
          icon={UserIcon}
          color="blue"
        />
        <StatCard
          title="Total Teams"
          value={teams.length}
          subtitle="Registered teams"
          icon={TrophyIcon}
          color="green"
        />
        <StatCard
          title="Predictions Made"
          value={predictions.length}
          subtitle="AI predictions"
          icon={BoltIcon}
          color="purple"
        />
        <StatCard
          title="WebSocket Status"
          value={wsStatus}
          subtitle="Real-time connection"
          icon={ChartIcon}
          color={wsStatus === 'connected' ? 'green' : 'red'}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold mb-4">Performance Trends</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="performance" stroke="#3B82F6" strokeWidth={2} />
              <Line type="monotone" dataKey="goals" stroke="#10B981" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold mb-4">Team Chemistry</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={teamChemistryData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({name, value}) => `${name}: ${value}%`}
              >
                {teamChemistryData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Real-time Updates</h3>
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-medium ${
            wsStatus === 'connected' 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              wsStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'
            }`}></div>
            {wsStatus}
          </div>
        </div>
        <div className="space-y-2 max-h-40 overflow-y-auto">
          {wsMessages.length === 0 ? (
            <p className="text-gray-500 text-sm">No real-time updates yet...</p>
          ) : (
            wsMessages.map((msg, index) => (
              <div key={index} className="text-sm p-2 bg-gray-50 rounded">
                <span className="font-medium text-blue-600">{msg.type}</span>: {JSON.stringify(msg.message || msg.data || 'Connected', null, 0)}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );

  const PlayersContent = () => (
    <div className="space-y-6">
      {/* Search */}
      <div className="relative">
        <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
        <input
          type="text"
          placeholder="Search players..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>

      {/* Players Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredPlayers.map((player) => (
          <div
            key={player.id}
            className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 hover:shadow-lg transition-all cursor-pointer"
            onClick={() => setSelectedPlayer(player)}
          >
            <div className="flex items-center space-x-4">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full flex items-center justify-center text-white font-bold text-lg">
                {player.name.split(' ').map(n => n[0]).join('')}
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900">{player.name}</h3>
                <p className="text-sm text-gray-600">{player.team}</p>
                <p className="text-xs text-gray-500 capitalize">{player.position}</p>
              </div>
            </div>
            
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{player.goals}</div>
                <div className="text-xs text-gray-500">Goals</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{player.assists}</div>
                <div className="text-xs text-gray-500">Assists</div>
              </div>
            </div>
            
            <div className="mt-4">
              <div className="flex justify-between text-sm mb-1">
                <span>Pass Accuracy</span>
                <span>{player.pass_accuracy}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full"
                  style={{ width: `${player.pass_accuracy}%` }}
                ></div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Player Modal */}
      {selectedPlayer && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl p-6 max-w-md w-full">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h2 className="text-xl font-bold">{selectedPlayer.name}</h2>
                <p className="text-gray-600">{selectedPlayer.team}</p>
              </div>
              <button
                onClick={() => setSelectedPlayer(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 p-3 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">{selectedPlayer.goals}</div>
                  <div className="text-sm text-gray-600">Goals</div>
                </div>
                <div className="bg-green-50 p-3 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">{selectedPlayer.assists}</div>
                  <div className="text-sm text-gray-600">Assists</div>
                </div>
                <div className="bg-purple-50 p-3 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">{selectedPlayer.pass_accuracy}%</div>
                  <div className="text-sm text-gray-600">Pass Accuracy</div>
                </div>
                <div className="bg-orange-50 p-3 rounded-lg">
                  <div className="text-2xl font-bold text-orange-600">{selectedPlayer.minutes_played}</div>
                  <div className="text-sm text-gray-600">Minutes</div>
                </div>
              </div>
              
              <div className="bg-gray-50 p-3 rounded-lg">
                <div className="text-sm text-gray-600">Position</div>
                <div className="font-semibold capitalize">{selectedPlayer.position}</div>
              </div>
              
              <div className="bg-gray-50 p-3 rounded-lg">
                <div className="text-sm text-gray-600">Age</div>
                <div className="font-semibold">{selectedPlayer.age} years</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const PredictionsContent = () => (
    <div className="space-y-6">
      {/* Prediction Form */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h3 className="text-lg font-semibold mb-4">Make AI Prediction</h3>
        <form onSubmit={handlePrediction} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Player</label>
              <select
                value={predictionForm.player_id}
                onChange={(e) => setPredictionForm({...predictionForm, player_id: e.target.value})}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="">Select a player</option>
                {players.map((player) => (
                  <option key={player.id} value={player.id}>
                    {player.name} - {player.team}
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Goals</label>
              <input
                type="number"
                value={predictionForm.goals}
                onChange={(e) => setPredictionForm({...predictionForm, goals: parseInt(e.target.value) || 0})}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                min="0"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Assists</label>
              <input
                type="number"
                value={predictionForm.assists}
                onChange={(e) => setPredictionForm({...predictionForm, assists: parseInt(e.target.value) || 0})}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                min="0"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Pass Accuracy (%)</label>
              <input
                type="number"
                value={predictionForm.pass_accuracy}
                onChange={(e) => setPredictionForm({...predictionForm, pass_accuracy: parseFloat(e.target.value) || 0})}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                min="0"
                max="100"
                step="0.1"
                required
              />
            </div>
            
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">Minutes Played</label>
              <input
                type="number"
                value={predictionForm.minutes_played}
                onChange={(e) => setPredictionForm({...predictionForm, minutes_played: parseInt(e.target.value) || 0})}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                min="0"
                required
              />
            </div>
          </div>
          
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {loading ? (
              <>
                <RefreshIcon className="animate-spin" />
                <span>Generating Prediction...</span>
              </>
            ) : (
              <>
                <BoltIcon />
                <span>Generate AI Prediction</span>
              </>
            )}
          </button>
        </form>
      </div>

      {/* Predictions List */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h3 className="text-lg font-semibold mb-4">Recent Predictions</h3>
        {predictions.length === 0 ? (
          <p className="text-gray-500">No predictions yet. Make your first prediction above!</p>
        ) : (
          <div className="space-y-4">
            {predictions.map((prediction, index) => (
              <div key={prediction.prediction_id || index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-white ${
                      prediction.performance_score >= 8 ? 'bg-green-500' :
                      prediction.performance_score >= 6 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}>
                      {prediction.performance_score}
                    </div>
                    <div>
                      <div className="font-semibold">Performance Score: {prediction.performance_score}/10</div>
                      <div className="text-sm text-gray-600">Confidence: {(prediction.confidence * 100).toFixed(1)}%</div>
                      <div className="text-sm text-gray-500">Model: {prediction.model_used}</div>
                    </div>
                  </div>
                  <div className="text-xs text-gray-400">
                    ID: {prediction.prediction_id}
                  </div>
                </div>
                
                {prediction.insights && prediction.insights.length > 0 && (
                  <div className="bg-blue-50 rounded-lg p-3">
                    <div className="font-medium text-blue-900 mb-2">AI Insights:</div>
                    <div className="space-y-1">
                      {prediction.insights.map((insight, idx) => (
                        <div key={idx} className="text-sm text-blue-800">
                          {insight}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  const TeamsContent = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {teams.map((team) => (
          <div key={team.id} className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 hover:shadow-lg transition-shadow">
            <div className="flex items-center space-x-4 mb-4">
              <div className="w-16 h-16 bg-gradient-to-r from-green-400 to-blue-500 rounded-full flex items-center justify-center text-white font-bold text-lg">
                {team.name.split(' ').map(n => n[0]).join('').slice(0, 2)}
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">{team.name}</h3>
                <p className="text-sm text-gray-600">{team.league}</p>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Players</span>
                <span className="font-semibold">{team.players?.length || 0}</span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Chemistry Score</span>
                <span className={`font-semibold ${
                  team.chemistry_score >= 0.8 ? 'text-green-600' :
                  team.chemistry_score >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {(team.chemistry_score * 100).toFixed(1)}%
                </span>
              </div>
              
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full"
                  style={{ width: `${team.chemistry_score * 100}%` }}
                ></div>
              </div>
            </div>
            
            {team.players && team.players.length > 0 && (
              <div className="mt-4 pt-4 border-t border-gray-100">
                <div className="text-xs text-gray-500 mb-2">Top Players:</div>
                <div className="space-y-1">
                  {team.players.slice(0, 3).map((player, idx) => (
                    <div key={idx} className="text-sm text-gray-700">
                      {player.name} - {player.position}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: ChartIcon },
    { id: 'players', label: 'Players', icon: UserIcon },
    { id: 'predictions', label: 'AI Predictions', icon: BoltIcon },
    { id: 'teams', label: 'Teams', icon: TrophyIcon }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <TrophyIcon className="text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Sports Analytics</h1>
                <p className="text-sm text-gray-600">AI-powered performance insights</p>
              </div>
            </div>
            
            <button
              onClick={fetchData}
              disabled={loading}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              <RefreshIcon className={loading ? 'animate-spin' : ''} />
              <span>Refresh</span>
            </button>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <tab.icon />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {loading && activeTab === 'dashboard' && (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        )}
        
        {!loading && (
          <>
            {activeTab === 'dashboard' && <DashboardContent />}
            {activeTab === 'players' && <PlayersContent />}
            {activeTab === 'predictions' && <PredictionsContent />}
            {activeTab === 'teams' && <TeamsContent />}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              © 2024 Sports Analytics Dashboard. Powered by AI.
            </div>
            <div className="flex items-center space-x-4 text-sm text-gray-500">
              <span>API Status: Connected</span>
              <span className={`flex items-center space-x-1 ${
                wsStatus === 'connected' ? 'text-green-600' : 'text-red-600'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  wsStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span>WebSocket: {wsStatus}</span>
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default SportsAnalyticsDashboard;