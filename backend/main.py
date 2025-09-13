"""
AI Sports Coach FastAPI Backend
Complete implementation with voice coaching, AI analytics, and real-time updates
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any, Union
import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
import uuid
import io
import base64
import random
import math
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# ML and AI imports (simulated for demo - in production use actual models)
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Sports Coach Backend",
    description="AI-powered sports coaching with voice interaction and real-time analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# DATA MODELS
# =======================

class BiometricData(BaseModel):
    heart_rate: Optional[int] = Field(None, ge=40, le=220, description="Heart rate in BPM")
    rpe: Optional[int] = Field(None, ge=1, le=10, description="Rate of Perceived Exertion")
    sleep_hours: Optional[float] = Field(None, ge=0, le=24, description="Hours of sleep")
    stress_level: Optional[int] = Field(None, ge=1, le=10, description="Stress level")
    hydration_level: Optional[int] = Field(None, ge=1, le=10, description="Hydration level")
    body_weight: Optional[float] = Field(None, ge=30, le=300, description="Body weight in kg")

class ExerciseData(BaseModel):
    exercise_name: str
    sets: int = Field(..., ge=1, le=20)
    reps: int = Field(..., ge=1, le=100)
    weight: Optional[float] = Field(None, ge=0, le=1000, description="Weight in kg")
    duration_seconds: Optional[int] = Field(None, ge=1, le=7200, description="Duration in seconds")
    distance_meters: Optional[float] = Field(None, ge=0, le=100000, description="Distance in meters")

class WorkoutSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    sport_type: str = Field(..., description="e.g., strength, cardio, swimming, running")
    session_date: datetime = Field(default_factory=datetime.now)
    duration_minutes: int = Field(..., ge=1, le=300)
    intensity: int = Field(..., ge=1, le=10, description="Workout intensity")
    exercises: List[ExerciseData] = Field(default_factory=list)
    biometrics: BiometricData = Field(default_factory=BiometricData)
    notes: Optional[str] = None

class TrainingPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    plan_name: str
    target_goal: str
    duration_weeks: int = Field(..., ge=1, le=52)
    weekly_sessions: int = Field(..., ge=1, le=14)
    current_week: int = Field(default=1)
    exercises_per_week: Dict[str, List[Dict]] = Field(default_factory=dict)
    created_date: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

class PerformanceMetrics(BaseModel):
    overall_score: float = Field(..., ge=0, le=100)
    strength_score: float = Field(..., ge=0, le=100)
    endurance_score: float = Field(..., ge=0, le=100)
    speed_score: float = Field(..., ge=0, le=100)
    recovery_score: float = Field(..., ge=0, le=100)
    consistency_score: float = Field(..., ge=0, le=100)
    percentile_rank: float = Field(..., ge=0, le=100)
    trend_direction: str = Field(..., pattern=r"^(improving|declining|stable)$")

class InjuryPrediction(BaseModel):
    risk_level: str = Field(..., pattern=r"^(low|medium|high)$")
    risk_score: float = Field(..., ge=0, le=1)
    risk_factors: List[str]
    recommendations: List[str]
    recovery_days_needed: int = Field(..., ge=0, le=14)

class VoiceResponse(BaseModel):
    transcript: str
    response_text: str
    response_audio: Optional[str] = None
    confidence: float = Field(..., ge=0, le=1)
    intent: str
    extracted_data: Optional[Dict] = None

class SmartInsight(BaseModel):
    insight_type: str
    title: str
    message: str
    priority: str = Field(..., pattern=r"^(low|medium|high)$")
    action_items: List[str]
    created_date: datetime = Field(default_factory=datetime.now)

# =======================
# IN-MEMORY STORAGE
# =======================

class InMemoryStorage:
    def __init__(self):
        self.users: Dict[str, Dict] = {}
        self.sessions: Dict[str, List[WorkoutSession]] = defaultdict(list)
        self.training_plans: Dict[str, TrainingPlan] = {}
        self.performance_history: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.injury_predictions: Dict[str, List[InjuryPrediction]] = defaultdict(list)
        self.insights: Dict[str, List[SmartInsight]] = defaultdict(list)
        self.user_preferences: Dict[str, Dict] = {}
        
        # Initialize with demo data
        self._init_demo_data()
    
    def _init_demo_data(self):
        """Initialize with sample data for demonstration"""
        demo_user_id = "demo_user_123"
        
        # Demo user profile
        self.users[demo_user_id] = {
            "user_id": demo_user_id,
            "name": "Demo Athlete",
            "age": 28,
            "weight": 75.0,
            "height": 178,
            "sport": "strength_training",
            "experience_level": "intermediate",
            "created_date": datetime.now() - timedelta(days=30)
        }
        
        # Demo workout sessions
        for i in range(10):
            session = WorkoutSession(
                user_id=demo_user_id,
                sport_type="strength_training",
                session_date=datetime.now() - timedelta(days=i*3),
                duration_minutes=random.randint(45, 90),
                intensity=random.randint(6, 9),
                exercises=[
                    ExerciseData(
                        exercise_name="Squat",
                        sets=4,
                        reps=8,
                        weight=80 + random.randint(-10, 10)
                    ),
                    ExerciseData(
                        exercise_name="Bench Press",
                        sets=4,
                        reps=10,
                        weight=70 + random.randint(-5, 5)
                    )
                ],
                biometrics=BiometricData(
                    heart_rate=random.randint(140, 180),
                    rpe=random.randint(6, 9),
                    sleep_hours=random.uniform(6.5, 8.5),
                    stress_level=random.randint(2, 6),
                    hydration_level=random.randint(6, 10)
                )
            )
            self.sessions[demo_user_id].append(session)

# Global storage instance
storage = InMemoryStorage()

# =======================
# AI MODELS & ANALYTICS
# =======================

class AICoachingEngine:
    def __init__(self):
        self.injury_model = None
        self.performance_model = None
        self.scaler = None
        self._init_models()
    
    def _init_models(self):
        """Initialize ML models - in production, load pre-trained models"""
        if ML_AVAILABLE:
            # Create simple models for demo
            self.injury_model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.performance_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.scaler = StandardScaler()
            
            # Train with synthetic data
            X_demo = np.random.rand(100, 8)  # 8 features
            y_injury = np.random.choice([0, 1, 2], 100)  # 0=low, 1=medium, 2=high risk
            y_performance = np.random.rand(100) * 100  # Performance score 0-100
            
            self.injury_model.fit(X_demo, y_injury)
            self.performance_model.fit(X_demo, y_performance)
            self.scaler.fit(X_demo)
    
    def extract_features(self, user_id: str) -> np.ndarray:
        """Extract features from user data for ML models"""
        sessions = storage.sessions.get(user_id, [])
        
        if not sessions:
            return np.zeros(8)  # Default features
        
        recent_sessions = sessions[-5:]  # Last 5 sessions
        
        # Calculate aggregate features
        avg_intensity = np.mean([s.intensity for s in recent_sessions])
        avg_duration = np.mean([s.duration_minutes for s in recent_sessions])
        avg_heart_rate = np.mean([s.biometrics.heart_rate or 150 for s in recent_sessions])
        avg_rpe = np.mean([s.biometrics.rpe or 5 for s in recent_sessions])
        avg_sleep = np.mean([s.biometrics.sleep_hours or 7 for s in recent_sessions])
        avg_stress = np.mean([s.biometrics.stress_level or 3 for s in recent_sessions])
        training_frequency = len(sessions) / 30  # Sessions per day over last 30 days
        workload_trend = self._calculate_workload_trend(sessions)
        
        return np.array([
            avg_intensity, avg_duration, avg_heart_rate, avg_rpe,
            avg_sleep, avg_stress, training_frequency, workload_trend
        ]).reshape(1, -1)
    
    def _calculate_workload_trend(self, sessions: List[WorkoutSession]) -> float:
        """Calculate workload trend over recent sessions"""
        if len(sessions) < 3:
            return 0.0
        
        recent_workload = sum(s.intensity * s.duration_minutes for s in sessions[-3:])
        previous_workload = sum(s.intensity * s.duration_minutes for s in sessions[-6:-3]) if len(sessions) >= 6 else recent_workload
        
        if previous_workload == 0:
            return 0.0
        
        return (recent_workload - previous_workload) / previous_workload
    
    def predict_injury_risk(self, user_id: str) -> InjuryPrediction:
        """Predict injury risk for user"""
        features = self.extract_features(user_id)
        
        if ML_AVAILABLE and self.injury_model:
            features_scaled = self.scaler.transform(features)
            risk_class = self.injury_model.predict(features_scaled)[0]
            risk_proba = self.injury_model.predict_proba(features_scaled)[0]
            risk_score = max(risk_proba)
        else:
            # Fallback logic
            risk_class = random.randint(0, 2)
            risk_score = random.uniform(0.3, 0.9)
        
        risk_levels = ["low", "medium", "high"]
        risk_factors = []
        recommendations = []
        recovery_days = 0
        
        # Generate risk factors and recommendations based on data
        sessions = storage.sessions.get(user_id, [])
        if sessions:
            recent_session = sessions[-1]
            
            if recent_session.intensity > 8:
                risk_factors.append("High training intensity")
                recommendations.append("Consider reducing intensity by 10-15%")
            
            if recent_session.biometrics.sleep_hours and recent_session.biometrics.sleep_hours < 7:
                risk_factors.append("Insufficient sleep")
                recommendations.append("Aim for 7-9 hours of sleep per night")
            
            if recent_session.biometrics.stress_level and recent_session.biometrics.stress_level > 7:
                risk_factors.append("High stress levels")
                recommendations.append("Incorporate stress management techniques")
        
        if risk_class == 2:  # High risk
            recovery_days = random.randint(3, 7)
            recommendations.append("Take 3-7 days of active recovery")
        elif risk_class == 1:  # Medium risk
            recovery_days = random.randint(1, 3)
            recommendations.append("Consider 1-2 days of lighter training")
        
        if not risk_factors:
            risk_factors = ["Current training load within normal range"]
        if not recommendations:
            recommendations = ["Continue current training routine with regular monitoring"]
        
        return InjuryPrediction(
            risk_level=risk_levels[risk_class],
            risk_score=float(risk_score),
            risk_factors=risk_factors,
            recommendations=recommendations,
            recovery_days_needed=recovery_days
        )
    
    def calculate_performance_metrics(self, user_id: str) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        sessions = storage.sessions.get(user_id, [])
        
        if not sessions:
            return PerformanceMetrics(
                overall_score=50.0,
                strength_score=50.0,
                endurance_score=50.0,
                speed_score=50.0,
                recovery_score=50.0,
                consistency_score=50.0,
                percentile_rank=50.0,
                trend_direction="stable"
            )
        
        recent_sessions = sessions[-10:]  # Last 10 sessions
        
        # Calculate component scores
        strength_score = self._calculate_strength_score(recent_sessions)
        endurance_score = self._calculate_endurance_score(recent_sessions)
        speed_score = self._calculate_speed_score(recent_sessions)
        recovery_score = self._calculate_recovery_score(recent_sessions)
        consistency_score = self._calculate_consistency_score(sessions)
        
        # Overall score is weighted average
        overall_score = (
            strength_score * 0.25 +
            endurance_score * 0.25 +
            speed_score * 0.2 +
            recovery_score * 0.15 +
            consistency_score * 0.15
        )
        
        # Calculate trend
        trend = self._calculate_trend(sessions)
        
        # Percentile rank (simulated)
        percentile_rank = min(95, overall_score + random.uniform(-10, 15))
        
        return PerformanceMetrics(
            overall_score=round(overall_score, 1),
            strength_score=round(strength_score, 1),
            endurance_score=round(endurance_score, 1),
            speed_score=round(speed_score, 1),
            recovery_score=round(recovery_score, 1),
            consistency_score=round(consistency_score, 1),
            percentile_rank=round(percentile_rank, 1),
            trend_direction=trend
        )
    
    def _calculate_strength_score(self, sessions: List[WorkoutSession]) -> float:
        """Calculate strength score based on progressive overload"""
        if not sessions:
            return 50.0
        
        strength_sessions = [s for s in sessions if s.sport_type in ["strength_training", "weightlifting"]]
        if not strength_sessions:
            return 60.0
        
        # Look for progressive overload in exercises
        total_volume = 0
        session_count = 0
        
        for session in strength_sessions:
            session_volume = 0
            for exercise in session.exercises:
                if exercise.weight and exercise.sets and exercise.reps:
                    session_volume += exercise.weight * exercise.sets * exercise.reps
            
            if session_volume > 0:
                total_volume += session_volume
                session_count += 1
        
        if session_count == 0:
            return 60.0
        
        avg_volume = total_volume / session_count
        
        # Normalize to 0-100 scale (arbitrary scaling for demo)
        score = min(100, (avg_volume / 10000) * 100)
        return max(30, score)  # Minimum score of 30
    
    def _calculate_endurance_score(self, sessions: List[WorkoutSession]) -> float:
        """Calculate endurance score based on cardiovascular metrics"""
        if not sessions:
            return 50.0
        
        cardio_sessions = [s for s in sessions if s.sport_type in ["cardio", "running", "cycling", "swimming"]]
        
        if not cardio_sessions:
            # Use heart rate data from strength sessions
            hr_data = [s.biometrics.heart_rate for s in sessions if s.biometrics.heart_rate]
            if hr_data:
                avg_hr = np.mean(hr_data)
                # Lower heart rate during exercise = better fitness (simplified)
                score = max(30, 100 - (avg_hr - 120) / 2)
                return min(100, score)
            return 60.0
        
        # Calculate based on duration and intensity
        endurance_score = 0
        for session in cardio_sessions:
            duration_score = min(session.duration_minutes / 60 * 50, 50)  # Max 50 points for duration
            intensity_score = session.intensity * 5  # Max 50 points for intensity
            endurance_score += duration_score + intensity_score
        
        avg_endurance = endurance_score / len(cardio_sessions)
        return min(100, max(30, avg_endurance))
    
    def _calculate_speed_score(self, sessions: List[WorkoutSession]) -> float:
        """Calculate speed score based on exercise performance"""
        # Simplified speed calculation
        speed_indicators = []
        
        for session in sessions:
            for exercise in session.exercises:
                if exercise.distance_meters and exercise.duration_seconds:
                    speed = exercise.distance_meters / exercise.duration_seconds
                    speed_indicators.append(speed * 10)  # Scale up
        
        if speed_indicators:
            avg_speed = np.mean(speed_indicators)
            return min(100, max(30, avg_speed))
        
        # Fallback based on intensity
        intensities = [s.intensity for s in sessions]
        if intensities:
            avg_intensity = np.mean(intensities)
            return avg_intensity * 10
        
        return 65.0
    
    def _calculate_recovery_score(self, sessions: List[WorkoutSession]) -> float:
        """Calculate recovery score based on sleep, stress, and RPE"""
        recovery_indicators = []
        
        for session in sessions:
            session_recovery = 0
            indicators_count = 0
            
            if session.biometrics.sleep_hours:
                sleep_score = min(session.biometrics.sleep_hours / 8 * 100, 100)
                session_recovery += sleep_score
                indicators_count += 1
            
            if session.biometrics.stress_level:
                stress_score = (10 - session.biometrics.stress_level) * 10
                session_recovery += stress_score
                indicators_count += 1
            
            if session.biometrics.rpe:
                rpe_score = (10 - session.biometrics.rpe) * 10
                session_recovery += rpe_score
                indicators_count += 1
            
            if indicators_count > 0:
                recovery_indicators.append(session_recovery / indicators_count)
        
        if recovery_indicators:
            return max(30, min(100, np.mean(recovery_indicators)))
        
        return 70.0
    
    def _calculate_consistency_score(self, sessions: List[WorkoutSession]) -> float:
        """Calculate consistency score based on training frequency"""
        if len(sessions) < 2:
            return 30.0
        
        # Calculate days between sessions
        sorted_sessions = sorted(sessions, key=lambda x: x.session_date)
        gaps = []
        
        for i in range(1, len(sorted_sessions)):
            gap = (sorted_sessions[i].session_date - sorted_sessions[i-1].session_date).days
            gaps.append(gap)
        
        if not gaps:
            return 50.0
        
        avg_gap = np.mean(gaps)
        consistency_score = max(30, 100 - (avg_gap - 2) * 10)  # Optimal gap is 2 days
        
        return min(100, consistency_score)
    
    def _calculate_trend(self, sessions: List[WorkoutSession]) -> str:
        """Calculate performance trend"""
        if len(sessions) < 4:
            return "stable"
        
        recent_sessions = sessions[-4:]
        older_sessions = sessions[-8:-4] if len(sessions) >= 8 else sessions[:-4]
        
        if not older_sessions:
            return "stable"
        
        recent_avg_intensity = np.mean([s.intensity for s in recent_sessions])
        older_avg_intensity = np.mean([s.intensity for s in older_sessions])
        
        diff = recent_avg_intensity - older_avg_intensity
        
        if diff > 0.5:
            return "improving"
        elif diff < -0.5:
            return "declining"
        else:
            return "stable"
    
    def generate_training_plan(self, user_id: str, goal: str, weeks: int) -> TrainingPlan:
        """Generate adaptive training plan based on user data and goals"""
        user_data = storage.users.get(user_id, {})
        sessions = storage.sessions.get(user_id, [])
        
        # Analyze current performance level
        if sessions:
            recent_sessions = sessions[-5:]
            avg_intensity = np.mean([s.intensity for s in recent_sessions])
            preferred_sports = list(set([s.sport_type for s in recent_sessions]))
        else:
            avg_intensity = 5
            preferred_sports = ["strength_training"]
        
        # Generate weekly exercise plans
        exercises_per_week = {}
        
        for week in range(1, weeks + 1):
            week_key = f"week_{week}"
            
            # Progressive overload
            intensity_modifier = 1.0 + (week - 1) * 0.05  # 5% increase per week
            
            weekly_plan = []
            
            if "strength" in goal.lower() or "strength_training" in preferred_sports:
                weekly_plan.extend([
                    {
                        "day": 1,
                        "type": "strength_training",
                        "exercises": [
                            {"name": "Squat", "sets": 4, "reps": 8, "intensity": min(10, int(avg_intensity * intensity_modifier))},
                            {"name": "Bench Press", "sets": 4, "reps": 10, "intensity": min(10, int(avg_intensity * intensity_modifier))},
                            {"name": "Deadlift", "sets": 3, "reps": 6, "intensity": min(10, int(avg_intensity * intensity_modifier))}
                        ]
                    },
                    {
                        "day": 3,
                        "type": "strength_training",
                        "exercises": [
                            {"name": "Pull-ups", "sets": 3, "reps": 10, "intensity": min(10, int(avg_intensity * intensity_modifier))},
                            {"name": "Overhead Press", "sets": 4, "reps": 8, "intensity": min(10, int(avg_intensity * intensity_modifier))},
                            {"name": "Rows", "sets": 4, "reps": 10, "intensity": min(10, int(avg_intensity * intensity_modifier))}
                        ]
                    }
                ])
            
            if "cardio" in goal.lower() or "endurance" in goal.lower():
                weekly_plan.append({
                    "day": 2,
                    "type": "cardio",
                    "exercises": [
                        {"name": "Running", "duration": 30 + week * 2, "intensity": min(10, int(avg_intensity * intensity_modifier))}
                    ]
                })
                
                weekly_plan.append({
                    "day": 5,
                    "type": "cardio",
                    "exercises": [
                        {"name": "Cycling", "duration": 45 + week * 3, "intensity": min(10, int(avg_intensity * intensity_modifier))}
                    ]
                })
            
            exercises_per_week[week_key] = weekly_plan
        
        return TrainingPlan(
            user_id=user_id,
            plan_name=f"{goal.title()} Training Plan",
            target_goal=goal,
            duration_weeks=weeks,
            weekly_sessions=len(exercises_per_week.get("week_1", [])),
            exercises_per_week=exercises_per_week
        )
    
    def generate_smart_insights(self, user_id: str) -> List[SmartInsight]:
        """Generate AI-powered insights for the user"""
        insights = []
        sessions = storage.sessions.get(user_id, [])
        
        if not sessions:
            insights.append(SmartInsight(
                insight_type="welcome",
                title="Welcome to AI Sports Coach!",
                message="Start logging your workouts to receive personalized insights and coaching.",
                priority="medium",
                action_items=["Log your first workout", "Set your fitness goals"]
            ))
            return insights
        
        # Recent performance insight
        recent_sessions = sessions[-5:]
        if len(recent_sessions) >= 3:
            intensities = [s.intensity for s in recent_sessions]
            avg_intensity = np.mean(intensities)
            
            if avg_intensity > 8:
                insights.append(SmartInsight(
                    insight_type="performance",
                    title="High Intensity Training Detected",
                    message=f"Your average training intensity is {avg_intensity:.1f}/10. Consider incorporating recovery sessions.",
                    priority="high",
                    action_items=["Schedule a recovery day", "Monitor fatigue levels", "Reduce intensity by 10-15%"]
                ))
            elif avg_intensity < 5:
                insights.append(SmartInsight(
                    insight_type="performance",
                    title="Training Intensity Opportunity",
                    message=f"Your recent training intensity is {avg_intensity:.1f}/10. You may benefit from increased intensity.",
                    priority="medium",
                    action_items=["Gradually increase workout intensity", "Set progressive goals"]
                ))
        
        # Sleep insight
        sleep_data = [s.biometrics.sleep_hours for s in recent_sessions if s.biometrics.sleep_hours]
        if sleep_data:
            avg_sleep = np.mean(sleep_data)
            if avg_sleep < 7:
                insights.append(SmartInsight(
                    insight_type="recovery",
                    title="Sleep Optimization Needed",
                    message=f"Your average sleep is {avg_sleep:.1f} hours. Aim for 7-9 hours for optimal recovery.",
                    priority="high",
                    action_items=["Establish consistent sleep schedule", "Create bedtime routine", "Limit screen time before bed"]
                ))
        
        # Consistency insight
        if len(sessions) >= 5:
            sorted_sessions = sorted(sessions, key=lambda x: x.session_date)
            recent_dates = [s.session_date for s in sorted_sessions[-5:]]
            date_gaps = [(recent_dates[i] - recent_dates[i-1]).days for i in range(1, len(recent_dates))]
            
            if any(gap > 7 for gap in date_gaps):
                insights.append(SmartInsight(
                    insight_type="consistency",
                    title="Training Consistency Alert",
                    message="You've had gaps of more than a week between sessions. Consistency is key for progress.",
                    priority="medium",
                    action_items=["Set weekly training schedule", "Start with shorter, frequent sessions", "Use calendar reminders"]
                ))
        
        return insights

# Initialize AI engine
ai_engine = AICoachingEngine()

# =======================
# WEBSOCKET MANAGER
# =======================

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket connected for user: {user_id}")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        logger.info(f"WebSocket disconnected for user: {user_id}")
    
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
                self.disconnect(user_id)
    
    async def send_json_message(self, data: dict, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(data)
            except Exception as e:
                logger.error(f"Error sending JSON to {user_id}: {e}")
                self.disconnect(user_id)

websocket_manager = WebSocketManager()

# =======================
# VOICE PROCESSING
# =======================

class VoiceProcessor:
    def __init__(self):
        self.intent_patterns = {
            "performance": ["performance", "metrics", "stats", "score", "how am i doing", "progress"],
            "workout_plan": ["workout", "plan", "exercise", "training", "routine", "what should i do"],
            "injury_risk": ["injury", "risk", "pain", "hurt", "sore", "recovery"],
            "recovery": ["recovery", "rest", "sleep", "tired", "fatigue", "how is my recovery"],
            "general_question": ["help", "what", "how", "when", "why", "tell me"]
        }
    
    def process_voice_command(self, text: str, user_id: str) -> VoiceResponse:
        """Process voice command and generate response"""
        text_lower = text.lower()
        
        # Detect intent
        intent = self._detect_intent(text_lower)
        
        # Generate response based on intent
        response_text, extracted_data = self._generate_response(intent, text_lower, user_id)
        
        # Simulate confidence (in production, use actual ASR confidence)
        confidence = random.uniform(0.85, 0.98)
        
        return VoiceResponse(
            transcript=text,
            response_text=response_text,
            response_audio=None,  # In production, use TTS
            confidence=confidence,
            intent=intent,
            extracted_data=extracted_data
        )
    
    def _detect_intent(self, text: str) -> str:
        """Detect user intent from text"""
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in text for pattern in patterns):
                return intent
        return "general_question"
    
    def _generate_response(self, intent: str, text: str, user_id: str) -> tuple:
        """Generate appropriate response based on intent"""
        extracted_data = {}
        
        if intent == "performance":
            metrics = ai_engine.calculate_performance_metrics(user_id)
            response_text = f"Your overall performance score is {metrics.overall_score}%. "
            response_text += f"Your strength score is {metrics.strength_score}%, "
            response_text += f"endurance is {metrics.endurance_score}%, "
            response_text += f"and recovery is {metrics.recovery_score}%. "
            response_text += f"Your performance trend is {metrics.trend_direction}."
            extracted_data = {"metrics": metrics.dict()}
            
        elif intent == "workout_plan":
            sessions = storage.sessions.get(user_id, [])
            if sessions:
                recent_session = sessions[-1]
                response_text = f"Based on your recent {recent_session.sport_type} session, "
                response_text += "I recommend focusing on progressive overload. "
                response_text += "Try increasing weight by 2.5kg or adding one extra rep per set. "
                response_text += "Don't forget to include 5-10 minutes of warm-up and cool-down."
            else:
                response_text = "I recommend starting with a balanced routine: "
                response_text += "3 strength training sessions and 2 cardio sessions per week. "
                response_text += "Begin with bodyweight exercises and gradually add weights."
            
        elif intent == "injury_risk":
            prediction = ai_engine.predict_injury_risk(user_id)
            response_text = f"Your current injury risk is {prediction.risk_level}. "
            if prediction.risk_level == "high":
                response_text += "I strongly recommend taking a few days of active recovery. "
            response_text += f"Main recommendations: {', '.join(prediction.recommendations[:2])}"
            extracted_data = {"injury_prediction": prediction.dict()}
            
        elif intent == "recovery":
            sessions = storage.sessions.get(user_id, [])
            if sessions and sessions[-1].biometrics.sleep_hours:
                sleep = sessions[-1].biometrics.sleep_hours
                response_text = f"Based on your {sleep} hours of sleep, "
                if sleep >= 7:
                    response_text += "your recovery looks good! You're ready for your next workout."
                else:
                    response_text += "you might need more rest. Consider a lighter training day."
            else:
                response_text = "For optimal recovery, aim for 7-9 hours of sleep, "
                response_text += "stay hydrated, and include rest days in your training schedule."
                
        else:  # general_question
            response_text = "I'm your AI sports coach! I can help you track performance, "
            response_text += "suggest workouts, assess injury risk, and provide recovery guidance. "
            response_text += "Try asking about your performance metrics or workout recommendations."
        
        return response_text, extracted_data

voice_processor = VoiceProcessor()

# =======================
# API ENDPOINTS
# =======================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Sports Coach Backend is running!", "version": "1.0.0"}

@app.post("/training")
async def submit_training_session(session: WorkoutSession):
    """Submit workout session data and get updated analysis"""
    try:
        # Store session
        storage.sessions[session.user_id].append(session)
        
        # Generate updated metrics
        performance_metrics = ai_engine.calculate_performance_metrics(session.user_id)
        injury_prediction = ai_engine.predict_injury_risk(session.user_id)
        insights = ai_engine.generate_smart_insights(session.user_id)
        
        # Store metrics
        storage.performance_history[session.user_id].append(performance_metrics)
        storage.injury_predictions[session.user_id].append(injury_prediction)
        
        # Generate updated training plan
        training_plan = ai_engine.generate_training_plan(
            session.user_id, 
            "strength_and_conditioning", 
            12
        )
        storage.training_plans[session.user_id] = training_plan
        
        response_data = {
            "session_id": session.session_id,
            "status": "success",
            "performance_metrics": performance_metrics.dict(),
            "injury_prediction": injury_prediction.dict(),
            "updated_plan": training_plan.dict(),
            "insights": [insight.dict() for insight in insights[-3:]],  # Latest 3 insights
            "recommendations": [
                "Great workout! Your intensity was well-balanced.",
                f"Next session: focus on {training_plan.exercises_per_week.get('week_1', [{}])[0].get('type', 'recovery')}",
                "Remember to stay hydrated and get adequate sleep."
            ]
        }
        
        # Send real-time update via WebSocket
        await websocket_manager.send_json_message({
            "type": "training_update",
            "data": response_data
        }, session.user_id)
        
        logger.info(f"Training session submitted for user {session.user_id}")
        return response_data
        
    except Exception as e:
        logger.error(f"Error submitting training session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
async def get_analytics(user_id: str = Query(..., description="User ID")):
    """Get performance analytics and trends"""
    try:
        sessions = storage.sessions.get(user_id, [])
        performance_history = storage.performance_history.get(user_id, [])
        
        if not sessions:
            raise HTTPException(status_code=404, detail="No training data found for user")
        
        # Calculate trends over time
        recent_sessions = sessions[-30:]  # Last 30 sessions
        
        # Weekly aggregation
        weekly_data = defaultdict(list)
        for session in recent_sessions:
            week = session.session_date.strftime("%Y-W%U")
            weekly_data[week].append({
                "intensity": session.intensity,
                "duration": session.duration_minutes,
                "sport_type": session.sport_type
            })
        
        # Performance trends
        performance_trends = []
        if performance_history:
            for i, metrics in enumerate(performance_history[-12:]):  # Last 12 records
                performance_trends.append({
                    "date": (datetime.now() - timedelta(days=(12-i)*7)).isoformat(),
                    "overall_score": metrics.overall_score,
                    "strength_score": metrics.strength_score,
                    "endurance_score": metrics.endurance_score,
                    "recovery_score": metrics.recovery_score
                })
        
        # Training volume analysis
        total_volume = sum(s.intensity * s.duration_minutes for s in sessions)
        avg_intensity = np.mean([s.intensity for s in sessions])
        total_sessions = len(sessions)
        
        # Sport type distribution
        sport_distribution = defaultdict(int)
        for session in sessions:
            sport_distribution[session.sport_type] += 1
        
        analytics_data = {
            "user_id": user_id,
            "summary": {
                "total_sessions": total_sessions,
                "total_training_volume": round(total_volume, 2),
                "average_intensity": round(avg_intensity, 2),
                "training_days": len(set(s.session_date.date() for s in sessions)),
                "most_common_sport": max(sport_distribution.items(), key=lambda x: x[1])[0] if sport_distribution else None
            },
            "weekly_trends": [
                {
                    "week": week,
                    "sessions": len(data),
                    "avg_intensity": round(np.mean([d["intensity"] for d in data]), 2),
                    "total_duration": sum(d["duration"] for d in data)
                }
                for week, data in sorted(weekly_data.items())
            ],
            "performance_trends": performance_trends,
            "sport_distribution": dict(sport_distribution),
            "current_metrics": performance_history[-1].dict() if performance_history else None
        }
        
        logger.info(f"Analytics retrieved for user {user_id}")
        return analytics_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
async def get_predictions(user_id: str = Query(..., description="User ID")):
    """Get injury risk and recovery predictions"""
    try:
        # Get current predictions
        injury_prediction = ai_engine.predict_injury_risk(user_id)
        
        # Calculate recovery forecast
        sessions = storage.sessions.get(user_id, [])
        recovery_forecast = {
            "current_fatigue_level": "medium",
            "recommended_recovery_days": injury_prediction.recovery_days_needed,
            "optimal_next_workout": "moderate intensity",
            "sleep_recommendation": "7-8 hours",
            "hydration_status": "good" if sessions and sessions[-1].biometrics.hydration_level and sessions[-1].biometrics.hydration_level >= 7 else "needs_attention"
        }
        
        # Weekly fatigue forecast
        fatigue_forecast = []
        base_fatigue = random.uniform(0.3, 0.7)
        
        for day in range(7):
            date = datetime.now() + timedelta(days=day)
            # Simulate fatigue based on training pattern
            if day in [1, 3, 5]:  # Training days
                fatigue_level = min(1.0, base_fatigue + random.uniform(0.1, 0.3))
            else:  # Recovery days
                fatigue_level = max(0.2, base_fatigue - random.uniform(0.1, 0.2))
            
            fatigue_forecast.append({
                "date": date.strftime("%Y-%m-%d"),
                "predicted_fatigue": round(fatigue_level, 2),
                "recommended_intensity": "low" if fatigue_level > 0.7 else "moderate" if fatigue_level > 0.4 else "high"
            })
        
        predictions_data = {
            "user_id": user_id,
            "injury_prediction": injury_prediction.dict(),
            "recovery_forecast": recovery_forecast,
            "fatigue_forecast": fatigue_forecast,
            "recommendations": {
                "immediate": injury_prediction.recommendations[:3],
                "weekly": [
                    "Monitor sleep quality and duration",
                    "Stay consistent with hydration",
                    "Include dynamic warm-ups",
                    "Schedule regular recovery sessions"
                ]
            }
        }
        
        # Store prediction
        storage.injury_predictions[user_id].append(injury_prediction)
        
        logger.info(f"Predictions generated for user {user_id}")
        return predictions_data
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance")
async def get_performance(user_id: str = Query(..., description="User ID")):
    """Get comprehensive performance analysis"""
    try:
        performance_metrics = ai_engine.calculate_performance_metrics(user_id)
        insights = ai_engine.generate_smart_insights(user_id)
        
        # Get performance history for comparison
        performance_history = storage.performance_history.get(user_id, [])
        
        # Calculate improvements
        improvements = {}
        if len(performance_history) >= 2:
            current = performance_history[-1]
            previous = performance_history[-2]
            
            improvements = {
                "overall_change": round(current.overall_score - previous.overall_score, 2),
                "strength_change": round(current.strength_score - previous.strength_score, 2),
                "endurance_change": round(current.endurance_score - previous.endurance_score, 2),
                "recovery_change": round(current.recovery_score - previous.recovery_score, 2)
            }
        
        # Generate specific recommendations
        recommendations = []
        
        if performance_metrics.strength_score < 70:
            recommendations.append({
                "category": "strength",
                "priority": "medium",
                "action": "Increase progressive overload in compound movements",
                "expected_improvement": "15-20% strength gains in 8 weeks"
            })
        
        if performance_metrics.endurance_score < 65:
            recommendations.append({
                "category": "endurance",
                "priority": "medium", 
                "action": "Add 2 cardio sessions per week, 30+ minutes each",
                "expected_improvement": "Improved VO2 max and cardiovascular health"
            })
        
        if performance_metrics.recovery_score < 60:
            recommendations.append({
                "category": "recovery",
                "priority": "high",
                "action": "Optimize sleep schedule and stress management",
                "expected_improvement": "Better workout quality and reduced injury risk"
            })
        
        performance_data = {
            "user_id": user_id,
            "current_metrics": performance_metrics.dict(),
            "improvements": improvements,
            "benchmark_comparison": {
                "percentile_rank": performance_metrics.percentile_rank,
                "comparison": "above average" if performance_metrics.percentile_rank > 60 else "average" if performance_metrics.percentile_rank > 40 else "below average",
                "areas_of_strength": [],
                "areas_for_improvement": []
            },
            "insights": [insight.dict() for insight in insights],
            "recommendations": recommendations,
            "next_assessment_date": (datetime.now() + timedelta(weeks=2)).isoformat()
        }
        
        # Determine areas of strength and improvement
        scores = {
            "strength": performance_metrics.strength_score,
            "endurance": performance_metrics.endurance_score,
            "speed": performance_metrics.speed_score,
            "recovery": performance_metrics.recovery_score,
            "consistency": performance_metrics.consistency_score
        }
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        performance_data["benchmark_comparison"]["areas_of_strength"] = [area for area, score in sorted_scores[:2]]
        performance_data["benchmark_comparison"]["areas_for_improvement"] = [area for area, score in sorted_scores[-2:]]
        
        logger.info(f"Performance analysis retrieved for user {user_id}")
        return performance_data
        
    except Exception as e:
        logger.error(f"Error retrieving performance data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice")
async def process_voice_command(
    user_id: str = Query(..., description="User ID"),
    audio_file: Optional[UploadFile] = File(None),
    text: Optional[str] = Query(None, description="Text command if no audio file")
):
    """Process voice command and return AI coaching response"""
    try:
        if not audio_file and not text:
            raise HTTPException(status_code=400, detail="Either audio file or text must be provided")
        
        # Process audio file (simplified - in production use Wav2Vec2)
        if audio_file:
            # Simulate speech-to-text processing
            audio_content = await audio_file.read()
            
            # Mock transcription (in production, use actual ASR)
            mock_transcriptions = [
                "Show my performance metrics",
                "What should I do for today's workout",
                "How is my recovery looking",
                "Am I at risk of injury",
                "Tell me about my progress"
            ]
            transcript = random.choice(mock_transcriptions)
        else:
            transcript = text
        
        # Process voice command
        voice_response = voice_processor.process_voice_command(transcript, user_id)
        
        # Send real-time update via WebSocket
        await websocket_manager.send_json_message({
            "type": "voice_response",
            "data": voice_response.dict()
        }, user_id)
        
        logger.info(f"Voice command processed for user {user_id}: {transcript}")
        return voice_response.dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing voice command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time coaching updates"""
    await websocket_manager.connect(websocket, user_id)
    
    try:
        # Send welcome message
        await websocket_manager.send_json_message({
            "type": "connection_established",
            "message": f"Connected to AI Sports Coach for user {user_id}",
            "timestamp": datetime.now().isoformat()
        }, user_id)
        
        # Start background coaching loop
        asyncio.create_task(real_time_coaching_loop(user_id))
        
        while True:
            # Listen for client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "request_update":
                # Send current metrics update
                performance_metrics = ai_engine.calculate_performance_metrics(user_id)
                await websocket_manager.send_json_message({
                    "type": "metrics_update",
                    "data": performance_metrics.dict(),
                    "timestamp": datetime.now().isoformat()
                }, user_id)
            
            elif message.get("type") == "voice_command":
                # Process voice command via WebSocket
                command_text = message.get("text", "")
                voice_response = voice_processor.process_voice_command(command_text, user_id)
                await websocket_manager.send_json_message({
                    "type": "voice_response",
                    "data": voice_response.dict()
                }, user_id)
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(user_id)
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        websocket_manager.disconnect(user_id)

async def real_time_coaching_loop(user_id: str):
    """Background loop for real-time coaching updates"""
    while user_id in websocket_manager.active_connections:
        try:
            # Generate periodic insights
            insights = ai_engine.generate_smart_insights(user_id)
            
            # Send coaching tips every 5 minutes
            coaching_tips = [
                "Remember to stay hydrated during your workout!",
                "Focus on proper form over heavy weights.",
                "Don't forget to warm up before intense exercises.",
                "Listen to your body - rest when you need it.",
                "Consistency is key to achieving your fitness goals."
            ]
            
            tip = random.choice(coaching_tips)
            
            await websocket_manager.send_json_message({
                "type": "coaching_tip",
                "message": tip,
                "timestamp": datetime.now().isoformat()
            }, user_id)
            
            # Wait 5 minutes before next update
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Error in coaching loop for user {user_id}: {e}")
            break

# =======================
# ADDITIONAL ENDPOINTS
# =======================

@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get user profile information"""
    user_data = storage.users.get(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Add computed statistics
    sessions = storage.sessions.get(user_id, [])
    user_data["statistics"] = {
        "total_sessions": len(sessions),
        "days_active": len(set(s.session_date.date() for s in sessions)),
        "favorite_sport": max(
            set(s.sport_type for s in sessions),
            key=lambda x: sum(1 for s in sessions if s.sport_type == x)
        ) if sessions else None,
        "member_since": user_data.get("created_date", datetime.now()).strftime("%Y-%m-%d")
    }
    
    return user_data

@app.post("/users/{user_id}/goals")
async def set_user_goals(user_id: str, goals: Dict[str, Any]):
    """Set user fitness goals"""
    if user_id not in storage.users:
        storage.users[user_id] = {"user_id": user_id}
    
    storage.users[user_id]["goals"] = goals
    storage.users[user_id]["goals_updated"] = datetime.now().isoformat()
    
    # Generate new training plan based on goals
    primary_goal = goals.get("primary_goal", "general_fitness")
    duration_weeks = goals.get("duration_weeks", 12)
    
    training_plan = ai_engine.generate_training_plan(user_id, primary_goal, duration_weeks)
    storage.training_plans[user_id] = training_plan
    
    return {
        "message": "Goals updated successfully",
        "training_plan": training_plan.dict()
    }

@app.get("/training-plans/{user_id}")
async def get_training_plan(user_id: str):
    """Get current training plan for user"""
    training_plan = storage.training_plans.get(user_id)
    if not training_plan:
        # Generate default plan
        training_plan = ai_engine.generate_training_plan(user_id, "general_fitness", 8)
        storage.training_plans[user_id] = training_plan
    
    return training_plan.dict()

@app.get("/insights/{user_id}")
async def get_insights(user_id: str):
    """Get AI-generated insights for user"""
    insights = ai_engine.generate_smart_insights(user_id)
    
    # Store insights
    storage.insights[user_id].extend(insights)
    
    # Keep only recent insights (last 20)
    storage.insights[user_id] = storage.insights[user_id][-20:]
    
    return {
        "insights": [insight.dict() for insight in insights],
        "total_insights": len(storage.insights[user_id])
    }

# =======================
# HEALTH CHECK & METRICS
# =======================

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "api": "running",
            "ml_engine": "available" if ML_AVAILABLE else "mock_mode",
            "websocket": f"{len(websocket_manager.active_connections)} active connections",
            "storage": f"{len(storage.users)} users, {sum(len(sessions) for sessions in storage.sessions.values())} sessions"
        }
    }

@app.get("/metrics")
async def get_system_metrics():
    """Get system metrics for monitoring"""
    total_sessions = sum(len(sessions) for sessions in storage.sessions.values())
    active_users = len([uid for uid, sessions in storage.sessions.items() if sessions and sessions[-1].session_date > datetime.now() - timedelta(days=7)])
    
    return {
        "total_users": len(storage.users),
        "total_sessions": total_sessions,
        "active_users_7d": active_users,
        "active_websockets": len(websocket_manager.active_connections),
        "ml_predictions_generated": sum(len(preds) for preds in storage.injury_predictions.values()),
        "training_plans_created": len(storage.training_plans),
        "system_uptime": "running",
        "timestamp": datetime.now().isoformat()
    }

# =======================
# ERROR HANDLERS
# =======================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# =======================
# STARTUP EVENT
# =======================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("AI Sports Coach Backend starting up...")
    logger.info(f"ML libraries available: {ML_AVAILABLE}")
    logger.info(f"Demo data initialized for user: demo_user_123")
    logger.info("Backend ready for connections!")

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )