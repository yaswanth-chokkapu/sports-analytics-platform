import React from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar, 
  PieChart, 
  Pie, 
  Cell 
} from 'recharts';
import { BarChart3, TrendingUp, Target, PieChart as PieChartIcon } from 'lucide-react';

const ChartsPage = ({ players, teams }) => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1 }
  };

  // Prepare data for charts
  const playerGoalsData = players.map(player => ({
    name: player.name.split(' ').pop(), // Last name only for cleaner display
    goals: player.goals,
    assists: player.assists,
    total: player.goals + player.assists
  }));

  const teamPointsData = teams.map(team => ({
    name: team.name.replace(/^(FC|CF)\s+/, '').split(' ')[0], // Simplified team names
    points: team.points,
    wins: team.wins,
    goals_for: team.goals_for,
    goals_against: team.goals_against
  }));

  const positionData = players.reduce((acc, player) => {
    const existing = acc.find(item => item.position === player.position);
    if (existing) {
      existing.count += 1;
      existing.avg_goals += player.goals;
    } else {
      acc.push({
        position: player.position,
        count: 1,
        avg_goals: player.goals
      });
    }
    return acc;
  }, []).map(item => ({
    ...item,
    avg_goals: Math.round(item.avg_goals / item.count)
  }));

  const playerComparisonData = players.slice(0, 5).map(player => ({
    name: player.name.split(' ').pop(),
    Goals: player.goals,
    Assists: player.assists,
    'Pass Accuracy': player.pass_accuracy,
    'Minutes (x100)': Math.round(player.minutes_played / 100),
    'Discipline': Math.max(100 - (player.yellow_cards * 10 + player.red_cards * 20), 0)
  }));

  const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={itemVariants} className="text-center py-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent mb-4">
          Performance Analytics
        </h1>
        <p className="text-gray-600 text-lg">
          Interactive charts and data visualizations for deep insights
        </p>
      </motion.div>

      {/* Player Goals vs Assists Chart */}
      <motion.div
        variants={itemVariants}
        className="bg-white rounded-xl shadow-lg border border-gray-100 p-6"
      >
        <div className="flex items-center mb-6">
          <BarChart3 className="w-6 h-6 text-blue-500 mr-2" />
          <h2 className="text-xl font-bold text-gray-900">Player Performance: Goals vs Assists</h2>
        </div>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={playerGoalsData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="name" 
                stroke="#666"
                fontSize={12}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis stroke="#666" fontSize={12} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Bar dataKey="goals" fill="#3b82f6" name="Goals" radius={[2, 2, 0, 0]} />
              <Bar dataKey="assists" fill="#10b981" name="Assists" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Team Points Trend */}
      <motion.div
        variants={itemVariants}
        className="bg-white rounded-xl shadow-lg border border-gray-100 p-6"
      >
        <div className="flex items-center mb-6">
          <TrendingUp className="w-6 h-6 text-green-500 mr-2" />
          <h2 className="text-xl font-bold text-gray-900">Team Performance: Points & Goals</h2>
        </div>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={teamPointsData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="name" 
                stroke="#666"
                fontSize={12}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis stroke="#666" fontSize={12} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="points" 
                stroke="#3b82f6" 
                strokeWidth={3}
                dot={{ fill: '#3b82f6', strokeWidth: 2, r: 6 }}
                name="Points"
              />
              <Line 
                type="monotone" 
                dataKey="goals_for" 
                stroke="#10b981" 
                strokeWidth={3}
                dot={{ fill: '#10b981', strokeWidth: 2, r: 6 }}
                name="Goals For"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Player Comparison Radar Chart */}
        <motion.div
          variants={itemVariants}
          className="bg-white rounded-xl shadow-lg border border-gray-100 p-6"
        >
          <div className="flex items-center mb-6">
            <Target className="w-6 h-6 text-purple-500 mr-2" />
            <h2 className="text-xl font-bold text-gray-900">Player Comparison</h2>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={playerComparisonData[0] ? [playerComparisonData[0]] : []}>
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis tick={{ fontSize: 12, fill: '#666' }} />
                <PolarRadiusAxis 
                  angle={90} 
                  domain={[0, 100]} 
                  tick={{ fontSize: 10, fill: '#666' }}
                />
                <Radar
                  name={playerComparisonData[0]?.name || 'Player'}
                  dataKey="Goals"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.2}
                  strokeWidth={2}
                />
                <Radar
                  name="Assists"
                  dataKey="Assists"
                  stroke="#10b981"
                  fill="#10b981"
                  fillOpacity={0.2}
                  strokeWidth={2}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px'
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Position Distribution */}
        <motion.div
          variants={itemVariants}
          className="bg-white rounded-xl shadow-lg border border-gray-100 p-6"
        >
          <div className="flex items-center mb-6">
            <PieChartIcon className="w-6 h-6 text-orange-500 mr-2" />
            <h2 className="text-xl font-bold text-gray-900">Position Distribution</h2>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={positionData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ position, count }) => `${position}: ${count}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {positionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Advanced Stats */}
      <motion.div
        variants={itemVariants}
        className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border border-gray-200 p-6"
      >
        <h2 className="text-xl font-bold text-gray-900 mb-6">Advanced Analytics</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {(players.reduce((sum, p) => sum + p.pass_accuracy, 0) / players.length).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Avg Pass Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {Math.round(players.reduce((sum, p) => sum + p.minutes_played, 0) / players.length).toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">Avg Minutes Played</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {(players.reduce((sum, p) => sum + p.goals, 0) / players.reduce((sum, p) => sum + p.minutes_played, 0) * 90).toFixed(3)}
            </div>
            <div className="text-sm text-gray-600">Goals per 90min</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">
              {players.reduce((sum, p) => sum + p.yellow_cards + p.red_cards, 0)}
            </div>
            <div className="text-sm text-gray-600">Total Cards</div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default ChartsPage;