import React, { useState, useEffect, useRef, createContext, useContext } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Mic, MicOff, Send, Activity, Target, Calendar, TrendingUp, AlertTriangle, CheckCircle, Users, Play, Pause } from 'lucide-react';

// Context for global state management
const AppContext = createContext();

const AppProvider = ({ children }) => {
  const [user, setUser] = useState({ id: 'demo_user_123', name: 'Demo Athlete' });
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [performanceData, setPerformanceData] = useState(null);
  const [trainingPlan, setTrainingPlan] = useState(null);
  const [websocket, setWebsocket] = useState(null);
  const [insights, setInsights] = useState([]);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Initialize WebSocket connection
    const ws = new WebSocket(`ws://localhost:8000/ws/${user.id}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setWebsocket(ws);
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWebsocket(null);
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [user.id]);

  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'training_update':
        setPerformanceData(data.data.performance_metrics);
        setInsights(data.data.insights);
        break;
      case 'voice_response':
        // Handle voice response updates
        break;
      case 'coaching_tip':
        setInsights(prev => [...prev, {
          insight_type: 'coaching_tip',
          title: 'Coaching Tip',
          message: data.message,
          priority: 'low'
        }]);
        break;
    }
  };

  const apiCall = async (endpoint, options = {}) => {
    try {
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API call failed:', error);
      throw error;
    }
  };

  return (
    <AppContext.Provider value={{
      user,
      currentPage,
      setCurrentPage,
      performanceData,
      setPerformanceData,
      trainingPlan,
      setTrainingPlan,
      websocket,
      insights,
      setInsights,
      predictions,
      setPredictions,
      loading,
      setLoading,
      apiCall
    }}>
      {children}
    </AppContext.Provider>
  );
};

// Navigation Component
const Navigation = () => {
  const { currentPage, setCurrentPage } = useContext(AppContext);

  const navItems = [
    { id: 'dashboard', icon: Activity, label: 'Dashboard' },
    { id: 'voice', icon: Mic, label: 'Voice Coach' },
    { id: 'training', icon: Calendar, label: 'Training Plan' },
    { id: 'performance', icon: TrendingUp, label: 'Performance' },
    { id: 'predictions', icon: Target, label: 'Predictions' }
  ];

  return (
    <nav className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 shadow-lg">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <h1 className="text-2xl font-bold">Elevate your GAME</h1>
        <div className="flex space-x-1">
          {navItems.map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              onClick={() => setCurrentPage(id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                currentPage === id
                  ? 'bg-white bg-opacity-20 shadow-lg'
                  : 'hover:bg-white hover:bg-opacity-10'
              }`}
            >
              <Icon size={18} />
              <span className="hidden md:inline">{label}</span>
            </button>
          ))}
        </div>
      </div>
    </nav>
  );
};

// Dashboard Component
const Dashboard = () => {
  const { user, performanceData, setPerformanceData, apiCall, setCurrentPage, insights } = useContext(AppContext);
  const [weeklyData, setWeeklyData] = useState([]);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      const [performance, analytics] = await Promise.all([
        apiCall(`/performance?user_id=${user.id}`),
        apiCall(`/analytics?user_id=${user.id}`)
      ]);
      
      setPerformanceData(performance.current_metrics);
      
      // Transform analytics data for chart
      if (analytics.weekly_trends) {
        const chartData = analytics.weekly_trends.slice(-6).map(week => ({
          week: week.week.split('-W')[1],
          intensity: week.avg_intensity,
          sessions: week.sessions,
          duration: week.total_duration
        }));
        setWeeklyData(chartData);
      }
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    }
  };

  const submitQuickSession = async () => {
    const quickSession = {
      user_id: user.id,
      sport_type: 'strength_training',
      duration_minutes: 60,
      intensity: 7,
      exercises: [
        {
          exercise_name: 'Squat',
          sets: 4,
          reps: 8,
          weight: 80
        }
      ],
      biometrics: {
        heart_rate: 150,
        rpe: 7,
        sleep_hours: 8.0
      }
    };

    try {
      await apiCall('/training', {
        method: 'POST',
        body: JSON.stringify(quickSession)
      });
      loadDashboardData();
    } catch (error) {
      console.error('Failed to submit session:', error);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-6 border border-blue-100">
        <div className="flex items-center space-x-4">
          <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-xl font-bold">
            {user.name.split(' ').map(n => n[0]).join('')}
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-800">Welcome back, {user.name}!</h2>
            <p className="text-gray-600">Ready to crush your fitness goals today?</p>
          </div>
        </div>
      </div>

      {/* Performance Snapshot */}
      {performanceData && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <MetricCard
            title="Overall Score"
            value={`${performanceData.overall_score}%`}
            color="blue"
            trend={performanceData.trend_direction}
          />
          <MetricCard
            title="Strength"
            value={`${performanceData.strength_score}%`}
            color="red"
          />
          <MetricCard
            title="Endurance"
            value={`${performanceData.endurance_score}%`}
            color="green"
          />
          <MetricCard
            title="Recovery"
            value={`${performanceData.recovery_score}%`}
            color="purple"
          />
        </div>
      )}

      {/* Weekly Progress Chart */}
      <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
        <h3 className="text-xl font-bold text-gray-800 mb-4">Weekly Progress</h3>
        {weeklyData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={weeklyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="week" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="intensity" stroke="#3B82F6" strokeWidth={3} />
              <Line type="monotone" dataKey="sessions" stroke="#10B981" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-64 flex items-center justify-center text-gray-500">
            No data available. Submit your first workout!
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <ActionButton
          title="Submit Session"
          description="Log your latest workout"
          icon={Activity}
          color="blue"
          onClick={submitQuickSession}
        />
        <ActionButton
          title="Check Performance"
          description="View detailed analytics"
          icon={TrendingUp}
          color="green"
          onClick={() => setCurrentPage('performance')}
        />
        <ActionButton
          title="Voice Coach"
          description="Get AI coaching advice"
          icon={Mic}
          color="purple"
          onClick={() => setCurrentPage('voice')}
        />
      </div>

      {/* Recent Insights */}
      {insights.length > 0 && (
        <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
          <h3 className="text-xl font-bold text-gray-800 mb-4">Latest Insights</h3>
          <div className="space-y-3">
            {insights.slice(0, 3).map((insight, index) => (
              <InsightCard key={index} insight={insight} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Voice Coaching Component
const VoiceCoaching = () => {
  const { user, apiCall, websocket } = useContext(AppContext);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [responses, setResponses] = useState([]);
  const [processing, setProcessing] = useState(false);

  const startRecording = () => {
    setIsRecording(true);
    // Simulate recording - in production, use Web Speech API
    setTimeout(() => {
      const mockTranscripts = [
        "Show my performance metrics",
        "What should I do for today's workout",
        "How is my recovery looking",
        "Am I at risk of injury",
        "Tell me about my progress this week"
      ];
      const randomTranscript = mockTranscripts[Math.floor(Math.random() * mockTranscripts.length)];
      setTranscript(randomTranscript);
      setIsRecording(false);
      processVoiceCommand(randomTranscript);
    }, 3000);
  };

  const processVoiceCommand = async (text) => {
    setProcessing(true);
    try {
      const response = await apiCall(`/voice?user_id=${user.id}&text=${encodeURIComponent(text)}`, {
        method: 'POST'
      });
      
      setResponses(prev => [...prev, {
        user: text,
        ai: response.response_text,
        timestamp: new Date().toLocaleTimeString()
      }]);
      setTranscript('');
    } catch (error) {
      console.error('Voice processing failed:', error);
    } finally {
      setProcessing(false);
    }
  };

  const sendTextCommand = () => {
    if (transcript.trim()) {
      processVoiceCommand(transcript);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100 min-h-96">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">Voice Coaching Session</h2>
        
        {/* Microphone Button */}
        <div className="flex justify-center mb-8">
          <button
            onClick={startRecording}
            disabled={isRecording || processing}
            className={`w-24 h-24 rounded-full flex items-center justify-center text-white text-3xl transition-all duration-300 transform ${
              isRecording 
                ? 'bg-red-500 scale-110 animate-pulse' 
                : processing
                ? 'bg-yellow-500 animate-spin'
                : 'bg-blue-500 hover:bg-blue-600 hover:scale-105'
            } shadow-lg`}
          >
            {processing ? (
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
            ) : isRecording ? (
              <MicOff />
            ) : (
              <Mic />
            )}
          </button>
        </div>
        
        <div className="text-center mb-6">
          {isRecording && <p className="text-red-500 font-medium">Listening...</p>}
          {processing && <p className="text-yellow-500 font-medium">Processing your request...</p>}
          {!isRecording && !processing && <p className="text-gray-600">Click the microphone to start</p>}
        </div>

        {/* Text Input Alternative */}
        <div className="flex space-x-2 mb-6">
          <input
            type="text"
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            placeholder="Or type your question here..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            onKeyPress={(e) => e.key === 'Enter' && sendTextCommand()}
          />
          <button
            onClick={sendTextCommand}
            disabled={!transcript.trim() || processing}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send size={18} />
          </button>
        </div>

        {/* Conversation History */}
        <div className="space-y-4 max-h-64 overflow-y-auto">
          {responses.map((response, index) => (
            <div key={index} className="space-y-2">
              {/* User Message */}
              <div className="flex justify-end">
                <div className="bg-blue-500 text-white px-4 py-2 rounded-2xl max-w-xs">
                  <p>{response.user}</p>
                  <span className="text-xs opacity-75">{response.timestamp}</span>
                </div>
              </div>
              
              {/* AI Response */}
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-800 px-4 py-2 rounded-2xl max-w-md">
                  <p>{response.ai}</p>
                  <span className="text-xs text-gray-500">{response.timestamp}</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {responses.length === 0 && (
          <div className="text-center text-gray-500 py-8">
            <p>Start a conversation with your AI coach!</p>
            <p className="text-sm mt-2">Try asking about your performance, workout recommendations, or recovery status.</p>
          </div>
        )}
      </div>
    </div>
  );
};

// Training Plan Component
const TrainingPlan = () => {
  const { user, trainingPlan, setTrainingPlan, apiCall } = useContext(AppContext);
  const [selectedWeek, setSelectedWeek] = useState(1);

  useEffect(() => {
    loadTrainingPlan();
  }, []);

  const loadTrainingPlan = async () => {
    try {
      const plan = await apiCall(`/training-plans/${user.id}`);
      setTrainingPlan(plan);
    } catch (error) {
      console.error('Failed to load training plan:', error);
    }
  };

  if (!trainingPlan) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p>Loading your personalized training plan...</p>
        </div>
      </div>
    );
  }

  const currentWeekPlan = trainingPlan.exercises_per_week[`week_${selectedWeek}`] || [];

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Plan Header */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-2xl p-6 border border-green-100">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">{trainingPlan.plan_name}</h2>
        <p className="text-gray-600 mb-4">Goal: {trainingPlan.target_goal}</p>
        <div className="flex items-center space-x-4">
          <span className="text-sm bg-green-100 text-green-800 px-3 py-1 rounded-full">
            Week {trainingPlan.current_week} of {trainingPlan.duration_weeks}
          </span>
          <span className="text-sm bg-blue-100 text-blue-800 px-3 py-1 rounded-full">
            {trainingPlan.weekly_sessions} sessions/week
          </span>
        </div>
      </div>

      {/* Week Selector */}
      <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
        <h3 className="text-lg font-bold text-gray-800 mb-4">Select Week</h3>
        <div className="flex flex-wrap gap-2">
          {Array.from({ length: trainingPlan.duration_weeks }, (_, i) => i + 1).map(week => (
            <button
              key={week}
              onClick={() => setSelectedWeek(week)}
              className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                selectedWeek === week
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Week {week}
            </button>
          ))}
        </div>
      </div>

      {/* Weekly Plan */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {currentWeekPlan.map((day, index) => (
          <WorkoutCard key={index} workout={day} />
        ))}
      </div>

      {currentWeekPlan.length === 0 && (
        <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100 text-center">
          <Calendar size={48} className="mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-bold text-gray-800 mb-2">No workouts scheduled</h3>
          <p className="text-gray-600">This week appears to be a rest week or the plan needs updating.</p>
        </div>
      )}
    </div>
  );
};

// Performance Check Component
const PerformanceCheck = () => {
  const { user, apiCall, setCurrentPage } = useContext(AppContext);
  const [formData, setFormData] = useState({
    endurance: 5,
    speed: 5,
    recovery_hours: 8,
    stress_level: 3,
    energy_level: 7
  });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSliderChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const submitAssessment = async () => {
    setLoading(true);
    try {
      // Submit a mock workout session with the assessment data
      const sessionData = {
        user_id: user.id,
        sport_type: 'assessment',
        duration_minutes: 30,
        intensity: Math.round((formData.endurance + formData.speed + formData.energy_level) / 3),
        exercises: [],
        biometrics: {
          sleep_hours: formData.recovery_hours,
          stress_level: formData.stress_level,
          rpe: 10 - formData.energy_level
        }
      };

      await apiCall('/training', {
        method: 'POST',
        body: JSON.stringify(sessionData)
      });

      const performance = await apiCall(`/performance?user_id=${user.id}`);
      setResults(performance.current_metrics);
    } catch (error) {
      console.error('Assessment submission failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Performance Assessment</h2>
        
        {!results ? (
          <div className="space-y-6">
            {/* Assessment Form */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <SliderInput
                label="Endurance Level"
                value={formData.endurance}
                onChange={(value) => handleSliderChange('endurance', value)}
                min={1}
                max={10}
                color="blue"
              />
              <SliderInput
                label="Speed Level"
                value={formData.speed}
                onChange={(value) => handleSliderChange('speed', value)}
                min={1}
                max={10}
                color="green"
              />
              <SliderInput
                label="Recovery Hours"
                value={formData.recovery_hours}
                onChange={(value) => handleSliderChange('recovery_hours', value)}
                min={4}
                max={12}
                color="purple"
              />
              <SliderInput
                label="Stress Level"
                value={formData.stress_level}
                onChange={(value) => handleSliderChange('stress_level', value)}
                min={1}
                max={10}
                color="red"
              />
              <div className="md:col-span-2">
                <SliderInput
                  label="Energy Level"
                  value={formData.energy_level}
                  onChange={(value) => handleSliderChange('energy_level', value)}
                  min={1}
                  max={10}
                  color="yellow"
                />
              </div>
            </div>

            <button
              onClick={submitAssessment}
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-500 to-purple-500 text-white py-3 px-6 rounded-lg font-medium hover:from-blue-600 hover:to-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              {loading ? 'Analyzing...' : 'Get Performance Analysis'}
            </button>
          </div>
        ) : (
          <PerformanceResults results={results} onReset={() => setResults(null)} />
        )}
      </div>
    </div>
  );
};

// Predictions & Insights Component
const PredictionsInsights = () => {
  const { user, predictions, setPredictions, insights, setInsights, apiCall } = useContext(AppContext);

  useEffect(() => {
    loadPredictions();
    loadInsights();
  }, []);

  const loadPredictions = async () => {
    try {
      const data = await apiCall(`/predictions?user_id=${user.id}`);
      setPredictions(data);
    } catch (error) {
      console.error('Failed to load predictions:', error);
    }
  };

  const loadInsights = async () => {
    try {
      const data = await apiCall(`/insights/${user.id}`);
      setInsights(data.insights);
    } catch (error) {
      console.error('Failed to load insights:', error);
    }
  };

  if (!predictions) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p>Loading predictions and insights...</p>
        </div>
      </div>
    );
  }

  const riskColor = predictions.injury_prediction.risk_level === 'high' ? 'red' : 
                   predictions.injury_prediction.risk_level === 'medium' ? 'yellow' : 'green';

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Injury Risk Assessment */}
      <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Injury Risk Assessment</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className={`w-24 h-24 mx-auto rounded-full flex items-center justify-center text-white text-2xl font-bold ${
              riskColor === 'red' ? 'bg-red-500' : 
              riskColor === 'yellow' ? 'bg-yellow-500' : 'bg-green-500'
            }`}>
              {Math.round(predictions.injury_prediction.risk_score * 100)}%
            </div>
            <h3 className="text-lg font-bold mt-2 capitalize">{predictions.injury_prediction.risk_level} Risk</h3>
          </div>
          
          <div>
            <h4 className="font-bold text-gray-800 mb-2">Risk Factors:</h4>
            <ul className="space-y-1">
              {predictions.injury_prediction.risk_factors.map((factor, index) => (
                <li key={index} className="text-sm text-gray-600 flex items-center">
                  <AlertTriangle size={16} className="mr-2 text-yellow-500" />
                  {factor}
                </li>
              ))}
            </ul>
          </div>
          
          <div>
            <h4 className="font-bold text-gray-800 mb-2">Recommendations:</h4>
            <ul className="space-y-1">
              {predictions.injury_prediction.recommendations.slice(0, 3).map((rec, index) => (
                <li key={index} className="text-sm text-gray-600 flex items-center">
                  <CheckCircle size={16} className="mr-2 text-green-500" />
                  {rec}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Fatigue Forecast */}
      <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
        <h3 className="text-xl font-bold text-gray-800 mb-4">7-Day Fatigue Forecast</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={predictions.fatigue_forecast}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tickFormatter={(date) => new Date(date).toLocaleDateString('en', { weekday: 'short' })} />
            <YAxis />
            <Tooltip 
              labelFormatter={(date) => new Date(date).toLocaleDateString()}
              formatter={(value, name) => [
                `${(value * 100).toFixed(0)}%`,
                name === 'predicted_fatigue' ? 'Fatigue Level' : name
              ]}
            />
            <Bar dataKey="predicted_fatigue" fill="#3B82F6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Smart Insights */}
      <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
        <h3 className="text-xl font-bold text-gray-800 mb-4">Smart Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {insights.map((insight, index) => (
            <InsightCard key={index} insight={insight} />
          ))}
        </div>
      </div>
    </div>
  );
};

// Helper Components
const MetricCard = ({ title, value, color, trend }) => {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    red: 'from-red-500 to-red-600',
    green: 'from-green-500 to-green-600',
    purple: 'from-purple-500 to-purple-600'
  };

  const trendIcon = trend === 'improving' ? '↗' : trend === 'declining' ? '↘' : '→';
  const trendColor = trend === 'improving' ? 'text-green-500' : trend === 'declining' ? 'text-red-500' : 'text-gray-500';

  return (
    <div className="bg-white rounded-xl p-4 shadow-lg border border-gray-100">
      <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${colorClasses[color]} flex items-center justify-center mb-3`}>
        <Activity size={24} className="text-white" />
      </div>
      <h3 className="text-sm font-medium text-gray-600">{title}</h3>
      <div className="flex items-center justify-between">
        <p className="text-2xl font-bold text-gray-800">{value}</p>
        {trend && (
          <span className={`text-sm font-medium ${trendColor}`}>
            {trendIcon}
          </span>
        )}
      </div>
    </div>
  );
};

const ActionButton = ({ title, description, icon: Icon, color, onClick }) => {
  const colorClasses = {
    blue: 'from-blue-50 to-blue-100 hover:from-blue-100 hover:to-blue-150 border-blue-200',
    green: 'from-green-50 to-green-100 hover:from-green-100 hover:to-green-150 border-green-200',
    purple: 'from-purple-50 to-purple-100 hover:from-purple-100 hover:to-purple-150 border-purple-200'
  };

  const iconColors = {
    blue: 'text-blue-600',
    green: 'text-green-600',
    purple: 'text-purple-600'
  };

  return (
    <button
      onClick={onClick}
      className={`bg-gradient-to-r ${colorClasses[color]} border rounded-xl p-6 text-left hover:shadow-lg transition-all duration-200 transform hover:scale-105`}
    >
      <Icon size={32} className={`${iconColors[color]} mb-4`} />
      <h3 className="text-lg font-bold text-gray-800 mb-2">{title}</h3>
      <p className="text-gray-600 text-sm">{description}</p>
    </button>
  );
};

const InsightCard = ({ insight }) => {
  const priorityColors = {
    high: 'border-red-200 bg-red-50',
    medium: 'border-yellow-200 bg-yellow-50',
    low: 'border-green-200 bg-green-50'
  };

  const priorityIcons = {
    high: <AlertTriangle size={16} className="text-red-500" />,
    medium: <Target size={16} className="text-yellow-500" />,
    low: <CheckCircle size={16} className="text-green-500" />
  };

  return (
    <div className={`border rounded-lg p-4 ${priorityColors[insight.priority]}`}>
      <div className="flex items-center space-x-2 mb-2">
        {priorityIcons[insight.priority]}
        <h4 className="font-bold text-gray-800">{insight.title}</h4>
      </div>
      <p className="text-gray-700 text-sm mb-3">{insight.message}</p>
      {insight.action_items && insight.action_items.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-gray-600 uppercase">Action Items:</p>
          {insight.action_items.slice(0, 2).map((item, index) => (
            <p key={index} className="text-xs text-gray-600">• {item}</p>
          ))}
        </div>
      )}
    </div>
  );
};

const WorkoutCard = ({ workout }) => {
  const typeColors = {
    strength_training: 'from-red-500 to-red-600',
    cardio: 'from-blue-500 to-blue-600',
    flexibility: 'from-green-500 to-green-600'
  };

  return (
    <div className="bg-white rounded-xl p-4 shadow-lg border border-gray-100">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-bold text-gray-800">Day {workout.day}</h3>
        <div className={`px-3 py-1 rounded-full text-white text-xs font-medium bg-gradient-to-r ${typeColors[workout.type] || 'from-gray-500 to-gray-600'}`}>
          {workout.type.replace('_', ' ')}
        </div>
      </div>
      
      <div className="space-y-2">
        {workout.exercises.map((exercise, index) => (
          <div key={index} className="bg-gray-50 rounded-lg p-3">
            <h4 className="font-medium text-gray-800">{exercise.name}</h4>
            <div className="flex items-center space-x-3 text-sm text-gray-600 mt-1">
              {exercise.sets && <span>{exercise.sets} sets</span>}
              {exercise.reps && <span>{exercise.reps} reps</span>}
              {exercise.duration && <span>{exercise.duration} min</span>}
              {exercise.intensity && (
                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs">
                  Intensity: {exercise.intensity}/10
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const SliderInput = ({ label, value, onChange, min, max, color }) => {
  const colorClasses = {
    blue: 'accent-blue-500',
    green: 'accent-green-500',
    purple: 'accent-purple-500',
    red: 'accent-red-500',
    yellow: 'accent-yellow-500'
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label className="font-medium text-gray-800">{label}</label>
        <span className="text-lg font-bold text-gray-600">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className={`w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200 ${colorClasses[color]}`}
      />
      <div className="flex justify-between text-xs text-gray-500">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
};

const PerformanceResults = ({ results, onReset }) => {
  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBackground = (score) => {
    if (score >= 80) return 'bg-green-100';
    if (score >= 60) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  const radarData = [
    { subject: 'Strength', value: results.strength_score },
    { subject: 'Endurance', value: results.endurance_score },
    { subject: 'Speed', value: results.speed_score },
    { subject: 'Recovery', value: results.recovery_score },
    { subject: 'Consistency', value: results.consistency_score }
  ];

  return (
    <div className="space-y-6">
      <div className="text-center">
        <div className={`inline-flex items-center justify-center w-24 h-24 rounded-full text-3xl font-bold ${getScoreBackground(results.overall_score)} ${getScoreColor(results.overall_score)}`}>
          {results.overall_score}%
        </div>
        <h3 className="text-2xl font-bold text-gray-800 mt-4">Overall Performance Score</h3>
        <p className="text-gray-600">You're in the {results.percentile_rank}th percentile</p>
      </div>

      {/* Performance Breakdown */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {[
          { label: 'Strength', value: results.strength_score },
          { label: 'Endurance', value: results.endurance_score },
          { label: 'Speed', value: results.speed_score },
          { label: 'Recovery', value: results.recovery_score },
          { label: 'Consistency', value: results.consistency_score }
        ].map((metric, index) => (
          <div key={index} className="text-center">
            <div className={`text-2xl font-bold ${getScoreColor(metric.value)}`}>
              {metric.value}%
            </div>
            <p className="text-sm text-gray-600">{metric.label}</p>
          </div>
        ))}
      </div>

      {/* Radar Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="subject" />
            <PolarRadiusAxis angle={90} domain={[0, 100]} />
            <Radar dataKey="value" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.3} strokeWidth={2} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Trend Information */}
      <div className={`p-4 rounded-lg ${
        results.trend_direction === 'improving' ? 'bg-green-100 text-green-800' :
        results.trend_direction === 'declining' ? 'bg-red-100 text-red-800' :
        'bg-gray-100 text-gray-800'
      }`}>
        <h4 className="font-bold mb-2">Performance Trend</h4>
        <p>
          Your performance is currently {results.trend_direction}. 
          {results.trend_direction === 'improving' && ' Keep up the excellent work!'}
          {results.trend_direction === 'declining' && ' Consider adjusting your training or recovery routine.'}
          {results.trend_direction === 'stable' && ' You\'re maintaining consistent performance levels.'}
        </p>
      </div>

      <button
        onClick={onReset}
        className="w-full bg-gray-500 text-white py-2 px-4 rounded-lg hover:bg-gray-600 transition-colors"
      >
        Take Another Assessment
      </button>
    </div>
  );
};

// Main App Component
const App = () => {
  return (
    <AppProvider>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <main className="py-6">
          <AppContent />
        </main>
      </div>
    </AppProvider>
  );
};

const AppContent = () => {
  const { currentPage } = useContext(AppContext);

  switch (currentPage) {
    case 'dashboard':
      return <Dashboard />;
    case 'voice':
      return <VoiceCoaching />;
    case 'training':
      return <TrainingPlan />;
    case 'performance':
      return <PerformanceCheck />;
    case 'predictions':
      return <PredictionsInsights />;
    default:
      return <Dashboard />;
  }
};

export default App;