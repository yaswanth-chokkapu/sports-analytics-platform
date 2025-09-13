import React from 'react';
import { motion } from 'framer-motion';
import { Trophy, Users, Target, TrendingUp, Star, Award } from 'lucide-react';

const Dashboard = ({ players, teams }) => {
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

  const topPlayers = players
    .sort((a, b) => (b.goals + b.assists) - (a.goals + a.assists))
    .slice(0, 3);

  const topTeams = teams
    .sort((a, b) => b.points - a.points)
    .slice(0, 3);

  const statsCards = [
    {
      title: 'Total Players',
      value: players.length,
      icon: Users,
      color: 'bg-blue-500',
      change: '+12%'
    },
    {
      title: 'Teams Tracked',
      value: teams.length,
      icon: Trophy,
      color: 'bg-green-500',
      change: '+8%'
    },
    {
      title: 'Avg Goals/Player',
      value: players.length ? (players.reduce((sum, p) => sum + p.goals, 0) / players.length).toFixed(1) : 0,
      icon: Target,
      color: 'bg-purple-500',
      change: '+15%'
    },
    {
      title: 'Avg Pass Accuracy',
      value: players.length ? `${(players.reduce((sum, p) => sum + p.pass_accuracy, 0) / players.length).toFixed(1)}%` : '0%',
      icon: TrendingUp,
      color: 'bg-orange-500',
      change: '+3%'
    }
  ];

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={itemVariants} className="text-center py-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
          Sports Performance Dashboard
        </h1>
        <p className="text-gray-600 text-lg">
          Real-time analytics and insights for professional sports performance
        </p>
      </motion.div>

      {/* Stats Cards */}
      <motion.div
        variants={itemVariants}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {statsCards.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <motion.div
              key={stat.title}
              whileHover={{ scale: 1.05, y: -5 }}
              className="bg-white p-6 rounded-xl shadow-lg border border-gray-100 hover:shadow-xl transition-all duration-300"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                  <p className="text-3xl font-bold text-gray-900">{stat.value}</p>
                  <p className="text-sm text-green-600 font-medium">{stat.change} from last month</p>
                </div>
                <div className={`${stat.color} p-3 rounded-lg`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
              </div>
            </motion.div>
          );
        })}
      </motion.div>

      {/* Top Performers Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Players */}
        <motion.div
          variants={itemVariants}
          className="bg-white rounded-xl shadow-lg border border-gray-100 p-6"
        >
          <div className="flex items-center mb-6">
            <Star className="w-6 h-6 text-yellow-500 mr-2" />
            <h2 className="text-xl font-bold text-gray-900">Top Performers</h2>
          </div>
          <div className="space-y-4">
            {topPlayers.map((player, index) => (
              <motion.div
                key={player.id}
                whileHover={{ x: 5 }}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold mr-3 ${
                    index === 0 ? 'bg-yellow-500' : index === 1 ? 'bg-gray-400' : 'bg-orange-500'
                  }`}>
                    {index + 1}
                  </div>
                  <div>
                    <p className="font-semibold text-gray-900">{player.name}</p>
                    <p className="text-sm text-gray-600">{player.team} • {player.position}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-bold text-gray-900">{player.goals + player.assists}</p>
                  <p className="text-xs text-gray-600">G+A</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Top Teams */}
        <motion.div
          variants={itemVariants}
          className="bg-white rounded-xl shadow-lg border border-gray-100 p-6"
        >
          <div className="flex items-center mb-6">
            <Award className="w-6 h-6 text-blue-500 mr-2" />
            <h2 className="text-xl font-bold text-gray-900">League Leaders</h2>
          </div>
          <div className="space-y-4">
            {topTeams.map((team, index) => (
              <motion.div
                key={team.id}
                whileHover={{ x: 5 }}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold mr-3 ${
                    index === 0 ? 'bg-yellow-500' : index === 1 ? 'bg-gray-400' : 'bg-orange-500'
                  }`}>
                    {index + 1}
                  </div>
                  <div>
                    <p className="font-semibold text-gray-900">{team.name}</p>
                    <p className="text-sm text-gray-600">{team.league}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-bold text-gray-900">{team.points}</p>
                  <p className="text-xs text-gray-600">Points</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Player Stats Overview */}
      <motion.div
        variants={itemVariants}
        className="bg-white rounded-xl shadow-lg border border-gray-100 p-6"
      >
        <h2 className="text-xl font-bold text-gray-900 mb-6">Player Statistics Overview</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Player</th>
                <th className="text-center py-3 px-4 font-semibold text-gray-900">Position</th>
                <th className="text-center py-3 px-4 font-semibold text-gray-900">Goals</th>
                <th className="text-center py-3 px-4 font-semibold text-gray-900">Assists</th>
                <th className="text-center py-3 px-4 font-semibold text-gray-900">Pass Acc.</th>
                <th className="text-center py-3 px-4 font-semibold text-gray-900">Minutes</th>
              </tr>
            </thead>
            <tbody>
              {players.map((player) => (
                <motion.tr
                  key={player.id}
                  whileHover={{ backgroundColor: '#f9fafb' }}
                  className="border-b border-gray-100 hover:bg-gray-50 transition-colors"
                >
                  <td className="py-3 px-4">
                    <div>
                      <p className="font-semibold text-gray-900">{player.name}</p>
                      <p className="text-sm text-gray-600">{player.team}</p>
                    </div>
                  </td>
                  <td className="text-center py-3 px-4 text-gray-700">{player.position}</td>
                  <td className="text-center py-3 px-4">
                    <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-sm font-semibold">
                      {player.goals}
                    </span>
                  </td>
                  <td className="text-center py-3 px-4">
                    <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-sm font-semibold">
                      {player.assists}
                    </span>
                  </td>
                  <td className="text-center py-3 px-4 text-gray-700">{player.pass_accuracy}%</td>
                  <td className="text-center py-3 px-4 text-gray-700">{player.minutes_played.toLocaleString()}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Recent Activity */}
      <motion.div
        variants={itemVariants}
        className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl border border-gray-200 p-6"
      >
        <h2 className="text-xl font-bold text-gray-900 mb-4">Platform Insights</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600">{players.reduce((sum, p) => sum + p.goals, 0)}</div>
            <div className="text-sm text-gray-600">Total Goals Scored</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-600">{players.reduce((sum, p) => sum + p.assists, 0)}</div>
            <div className="text-sm text-gray-600">Total Assists Made</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600">{teams.reduce((sum, t) => sum + t.wins, 0)}</div>
            <div className="text-sm text-gray-600">Total Wins</div>
          </div>
        </div>
      </motion.div>
    </motion.div>
    );
};
export default Dashboard;