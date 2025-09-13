import React, { useState } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import { Target, TrendingUp, Award, AlertCircle, Loader2, Zap } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const PredictionForm = () => {
  const [formData, setFormData] = useState({
    goals: 0,
    assists: 0,
    passes: 0,
    pass_accuracy: 0,
    minutes_played: 0,
    yellow_cards: 0,
    red_cards: 0
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, formData);
      setPrediction(response.data);
    } catch (err) {
      setError('Failed to get prediction. Please check your connection and try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      goals: 0,
      assists: 0,
      passes: 0,
      pass_accuracy: 0,
      minutes_played: 0,
      yellow_cards: 0,
      red_cards: 0
    });
    setPrediction(null);
    setError(null);
  };

  const fillSampleData = () => {
    setFormData({
      goals: 15,
      assists: 10,
      passes: 2500,
      pass_accuracy: 88.5,
      minutes_played: 2800,
      yellow_cards: 3,
      red_cards: 0
    });
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1 }
  };

  const getRatingColor = (rating) => {
    switch (rating) {
      case 'World Class': return 'text-purple-600 bg-purple-100';
      case 'Excellent': return 'text-green-600 bg-green-100';
      case 'Very Good': return 'text-blue-600 bg-blue-100';
      case 'Good': return 'text-yellow-600 bg-yellow-100';
      case 'Average': return 'text-orange-600 bg-orange-100';
      default: return 'text-red-600 bg-red-100';
    }
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="max-w-6xl mx-auto space-y-6"
    >
      {/* Header */}
      <motion.div variants={itemVariants} className="text-center py-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
          Performance Prediction Engine
        </h1>
        <p className="text-gray-600 text-lg">
          Enter player statistics to predict performance score and win probability
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <motion.div
          variants={itemVariants}
          className="bg-white rounded-xl shadow-lg border border-gray-100 p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <Target className="w-6 h-6 text-purple-500 mr-2" />
              <h2 className="text-xl font-bold text-gray-900">Player Statistics</h2>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={fillSampleData}
                className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
              >
                Sample Data
              </button>
              <button
                onClick={resetForm}
                className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
              >
                Reset
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Goals Scored
                </label>
                <input
                  type="number"
                  name="goals"
                  value={formData.goals}
                  onChange={handleInputChange}
                  min="0"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="e.g., 15"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Assists Made
                </label>
                <input
                  type="number"
                  name="assists"
                  value={formData.assists}
                  onChange={handleInputChange}
                  min="0"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="e.g., 10"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Total Passes
                </label>
                <input
                  type="number"
                  name="passes"
                  value={formData.passes}
                  onChange={handleInputChange}
                  min="0"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="e.g., 2500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Pass Accuracy (%)
                </label>
                <input
                  type="number"
                  name="pass_accuracy"
                  value={formData.pass_accuracy}
                  onChange={handleInputChange}
                  min="0"
                  max="100"
                  step="0.1"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="e.g., 88.5"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Minutes Played
                </label>
                <input
                  type="number"
                  name="minutes_played"
                  value={formData.minutes_played}
                  onChange={handleInputChange}
                  min="0"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="e.g., 2800"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Yellow Cards
                </label>
                <input
                  type="number"
                  name="yellow_cards"
                  value={formData.yellow_cards}
                  onChange={handleInputChange}
                  min="0"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="e.g., 3"
                />
              </div>

              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Red Cards
                </label>
                <input
                  type="number"
                  name="red_cards"
                  value={formData.red_cards}
                  onChange={handleInputChange}
                  min="0"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="e.g., 0"
                />
              </div>
            </div>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin mr-2" />
              ) : (
                <Zap className="w-5 h-5 mr-2" />
              )}
              {loading ? 'Analyzing...' : 'Predict Performance'}
            </motion.button>
          </form>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center"
            >
              <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
              <span className="text-red-700">{error}</span>
            </motion.div>
          )}
        </motion.div>

        {/* Results */}
        <motion.div
          variants={itemVariants}
          className="bg-white rounded-xl shadow-lg border border-gray-100 p-6"
        >
          <div className="flex items-center mb-6">
            <TrendingUp className="w-6 h-6 text-green-500 mr-2" />
            <h2 className="text-xl font-bold text-gray-900">Prediction Results</h2>
          </div>

          {prediction ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-6"
            >
              {/* Performance Score */}
              <div className="text-center p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border">
                <div className="text-4xl font-bold text-gray-900 mb-2">
                  {prediction.performance_score}/100
                </div>
                <div className="text-lg text-gray-600">Performance Score</div>
                <div className="mt-2">
                  <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getRatingColor(prediction.rating)}`}>
                    {prediction.rating}
                  </span>
                </div>
              </div>

              {/* Win Probability */}
              <div className="text-center p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border">
                <div className="text-4xl font-bold text-gray-900 mb-2">
                  {prediction.win_probability}%
                </div>
                <div className="text-lg text-gray-600">Win Probability</div>
              </div>

              {/* Insights */}
              <div>
                <div className="flex items-center mb-3">
                  <Award className="w-5 h-5 text-yellow-500 mr-2" />
                  <h3 className="font-semibold text-gray-900">Performance Insights</h3>
                </div>
                <div className="space-y-2">
                  {prediction.insights.map((insight, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="flex items-center p-3 bg-gray-50 rounded-lg"
                    >
                      <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                      <span className="text-gray-700">{insight}</span>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Performance Breakdown */}
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {((formData.goals + formData.assists) / Math.max(formData.minutes_played / 90, 1)).toFixed(2)}
                  </div>
                  <div className="text-sm text-gray-600">G+A per 90min</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {Math.max(100 - (formData.yellow_cards * 5 + formData.red_cards * 20), 0)}
                  </div>
                  <div className="text-sm text-gray-600">Discipline Score</div>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="text-center py-12">
              <Target className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500 text-lg">
                Enter player statistics and click "Predict Performance" to see AI-powered predictions
              </p>
              <p className="text-gray-400 text-sm mt-2">
                Our algorithm analyzes goals, assists, passing accuracy, and discipline to provide insights
              </p>
            </div>
          )}
        </motion.div>
      </div>

      {/* How It Works */}
      <motion.div
        variants={itemVariants}
        className="bg-gradient-to-r from-indigo-50 to-cyan-50 rounded-xl border border-gray-200 p-6"
      >
        <h2 className="text-xl font-bold text-gray-900 mb-4">How Our Prediction Engine Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-3">
              <Target className="w-6 h-6 text-white" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Statistical Analysis</h3>
            <p className="text-sm text-gray-600">
              We analyze goals, assists, passing accuracy, and playing time to evaluate offensive contribution
            </p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-3">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Performance Modeling</h3>
            <p className="text-sm text-gray-600">
              Advanced algorithms weight different statistics to calculate an overall performance score
            </p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-purple-500 rounded-full flex items-center justify-center mx-auto mb-3">
              <Award className="w-6 h-6 text-white" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Predictive Insights</h3>
            <p className="text-sm text-gray-600">
              Generate win probability and personalized insights based on the player's statistical profile
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default PredictionForm;