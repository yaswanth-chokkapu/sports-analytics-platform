from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import asyncio
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import logging
import hashlib
import hmac
import orjson
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Enhanced Data Models
class Position(str, Enum):
    GOALKEEPER = "goalkeeper"
    DEFENDER = "defender"
    MIDFIELDER = "midfielder"
    FORWARD = "forward"

class Player(BaseModel):
    id: int
    name: str
    position: Position
    team: str
    age: int
    goals: int = 0
    assists: int = 0
    pass_accuracy: float = 0.0
    minutes_played: int = 0
    injury_risk: float = 0.0
    market_value: float = 0.0
    performance_vector: List[float] = Field(default_factory=list)
    sentiment_score: float = 0.0

class Team(BaseModel):
    id: int
    name: str
    league: str
    players: List[Player] = Field(default_factory=list)
    chemistry_score: float = 0.0

class PredictionRequest(BaseModel):
    player_id: int
    goals: int
    assists: int
    pass_accuracy: float
    minutes_played: int
    
class PredictionResponse(BaseModel):
    performance_score: float
    confidence: float
    model_used: str
    insights: List[str] = Field(default_factory=list)
    prediction_id: str

class AnalyticsData(BaseModel):
    timestamp: datetime
    metric_name: str
    metric_value: float
    player_id: Optional[int] = None
    team_id: Optional[int] = None

# Mock Database (In production, use actual databases)
class MockDatabase:
    def __init__(self):
        self.players: Dict[int, Player] = {}
        self.teams: Dict[int, Team] = {}
        self.predictions: List[PredictionResponse] = []
        self.analytics: List[AnalyticsData] = []
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize with some mock data"""
        mock_players = [
            Player(
                id=1, name="Lionel Messi", position=Position.FORWARD, 
                team="PSG", age=36, goals=25, assists=15, 
                pass_accuracy=85.5, minutes_played=2500,
                market_value=50000000.0
            ),
            Player(
                id=2, name="Kevin De Bruyne", position=Position.MIDFIELDER,
                team="Manchester City", age=32, goals=8, assists=20,
                pass_accuracy=92.3, minutes_played=2800,
                market_value=75000000.0
            ),
            Player(
                id=3, name="Virgil van Dijk", position=Position.DEFENDER,
                team="Liverpool", age=32, goals=2, assists=3,
                pass_accuracy=89.7, minutes_played=3000,
                market_value=45000000.0
            )
        ]
        
        for player in mock_players:
            self.players[player.id] = player
        
        # Create mock teams
        teams_data = [
            {"id": 1, "name": "PSG", "league": "Ligue 1"},
            {"id": 2, "name": "Manchester City", "league": "Premier League"},
            {"id": 3, "name": "Liverpool", "league": "Premier League"}
        ]
        
        for team_data in teams_data:
            team_players = [p for p in self.players.values() if p.team == team_data["name"]]
            team = Team(**team_data, players=team_players)
            self.teams[team.id] = team

# AI Performance Predictor (Mock implementation)
class AIPerformancePredictor:
    def __init__(self):
        self.models = {
            'linear': 'mock_linear_model',
            'ensemble': 'mock_ensemble_model',
            'neural_network': 'mock_nn_model'
        }
        
    async def predict_performance(self, request: PredictionRequest) -> PredictionResponse:
        """Mock AI prediction with realistic logic"""
        
        # Simple scoring algorithm (replace with actual ML model)
        base_score = (
            request.goals * 0.4 +
            request.assists * 0.3 +
            (request.pass_accuracy / 100) * 0.2 +
            (request.minutes_played / 3000) * 0.1
        ) * 10
        
        # Add some randomness for realism
        import random
        noise = random.uniform(-0.5, 0.5)
        performance_score = max(0, min(10, base_score + noise))
        
        # Generate insights based on stats
        insights = self._generate_insights(request)
        
        prediction_id = f"pred_{int(datetime.now().timestamp())}"
        
        return PredictionResponse(
            performance_score=round(performance_score, 2),
            confidence=0.85,
            model_used='ensemble',
            insights=insights,
            prediction_id=prediction_id
        )
    
    def _generate_insights(self, request: PredictionRequest) -> List[str]:
        insights = []
        
        if request.goals > 20:
            insights.append("🔥 Elite goal scorer - Exceptional finishing ability")
        
        if request.assists > 15:
            insights.append("🎯 Creative playmaker - Outstanding assist record")
        
        if request.pass_accuracy > 90:
            insights.append("⚡ Technical excellence - Superior passing accuracy")
        
        if request.minutes_played > 2500:
            insights.append("💪 Reliable performer - High game time indicates fitness")
        
        if not insights:
            insights.append("📊 Balanced performance across all metrics")
            
        return insights

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.user_connections[user_id] = websocket
        logger.info(f"User {user_id} connected via WebSocket")
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id in self.user_connections:
            del self.user_connections[user_id]
        logger.info(f"User {user_id} disconnected")
    
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.user_connections:
            try:
                await self.user_connections[user_id].send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}: {e}")
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)

# Application Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Sports Analytics API...")
    
    # Initialize services
    app.state.db = MockDatabase()
    app.state.ai_predictor = AIPerformancePredictor()
    app.state.connection_manager = ConnectionManager()
    
    # Initialize cache (mock)
    app.state.cache = {}
    
    logger.info("✅ All services initialized successfully!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Sports Analytics API...")

# FastAPI App Initialization
app = FastAPI(
    title="Next-Gen Sports Performance Analytics API",
    version="2.0.0",
    description="Advanced AI-powered sports analytics with real-time processing",
    lifespan=lifespan
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Helper Functions
def get_cache_key(prefix: str, identifier: str) -> str:
    return f"{prefix}:{identifier}"

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Mock authentication - replace with real auth"""
    return {"user_id": "mock_user", "username": "test_user"}

# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """API Information"""
    return {
        "message": "Next-Gen Sports Performance Analytics API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "AI Performance Prediction",
            "Real-time WebSocket Updates", 
            "Player Analytics",
            "Team Chemistry Analysis",
            "Advanced Insights Generation"
        ],
        "endpoints": {
            "players": "/players",
            "predictions": "/predict",
            "analytics": "/analytics",
            "websocket": "/ws/{user_id}"
        }
    }

@app.get("/players", response_model=List[Player], tags=["Players"])
async def get_players(
    team: Optional[str] = None,
    position: Optional[Position] = None,
    limit: int = 50
):
    """Get all players with optional filtering"""
    
    # Check cache first
    cache_key = get_cache_key("players", f"{team}_{position}_{limit}")
    if cache_key in app.state.cache:
        return app.state.cache[cache_key]
    
    players = list(app.state.db.players.values())
    
    # Apply filters
    if team:
        players = [p for p in players if p.team.lower() == team.lower()]
    
    if position:
        players = [p for p in players if p.position == position]
    
    # Apply limit
    players = players[:limit]
    
    # Cache result
    app.state.cache[cache_key] = players
    
    logger.info(f"Retrieved {len(players)} players")
    return players

@app.get("/players/{player_id}", response_model=Player, tags=["Players"])
async def get_player(player_id: int):
    """Get specific player by ID"""
    
    if player_id not in app.state.db.players:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player with ID {player_id} not found"
        )
    
    return app.state.db.players[player_id]

@app.post("/players", response_model=Player, tags=["Players"])
async def create_player(player: Player, current_user: dict = Depends(get_current_user)):
    """Create a new player"""
    
    if player.id in app.state.db.players:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Player with ID {player.id} already exists"
        )
    
    app.state.db.players[player.id] = player
    
    # Clear cache
    app.state.cache.clear()
    
    logger.info(f"Created new player: {player.name}")
    return player

@app.get("/teams", response_model=List[Team], tags=["Teams"])
async def get_teams(league: Optional[str] = None):
    """Get all teams with optional league filtering"""
    
    teams = list(app.state.db.teams.values())
    
    if league:
        teams = [t for t in teams if t.league.lower() == league.lower()]
    
    return teams

@app.get("/teams/{team_id}", response_model=Team, tags=["Teams"])
async def get_team(team_id: int):
    """Get specific team by ID"""
    
    if team_id not in app.state.db.teams:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Team with ID {team_id} not found"
        )
    
    return app.state.db.teams[team_id]

@app.post("/predict", response_model=PredictionResponse, tags=["AI Predictions"])
async def predict_performance(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """AI-powered performance prediction"""
    
    try:
        # Validate player exists
        if request.player_id not in app.state.db.players:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Player with ID {request.player_id} not found"
            )
        
        # Make prediction
        prediction = await app.state.ai_predictor.predict_performance(request)
        
        # Store prediction
        app.state.db.predictions.append(prediction)
        
        # Background task for additional processing
        background_tasks.add_task(process_prediction_analytics, prediction, request)
        
        # Broadcast to WebSocket clients
        await app.state.connection_manager.broadcast(
            orjson.dumps({
                "type": "prediction",
                "data": prediction.dict()
            }).decode()
        )
        
        logger.info(f"Generated prediction {prediction.prediction_id} for player {request.player_id}")
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate prediction"
        )

@app.get("/predictions", response_model=List[PredictionResponse], tags=["AI Predictions"])
async def get_predictions(
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get recent predictions"""
    
    predictions = app.state.db.predictions[-limit:]
    return list(reversed(predictions))  # Most recent first

@app.get("/analytics/player/{player_id}/heatmap", tags=["Analytics"])
async def get_player_heatmap(player_id: int):
    """Generate player performance heatmap"""
    
    if player_id not in app.state.db.players:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player with ID {player_id} not found"
        )
    
    player = app.state.db.players[player_id]
    
    # Mock heatmap data based on player stats
    heatmap_data = {
        "player_id": player_id,
        "player_name": player.name,
        "positions": [
            {"x": 50 + (player.goals * 0.5), "y": 30, "intensity": min(1.0, player.goals / 30), "action": "goal"},
            {"x": 45, "y": 25 + (player.assists * 0.3), "intensity": min(1.0, player.assists / 20), "action": "assist"},
            {"x": 60, "y": 40, "intensity": player.pass_accuracy / 100, "action": "key_pass"}
        ],
        "generated_at": datetime.now().isoformat()
    }
    
    return heatmap_data

@app.get("/analytics/team/{team_id}/chemistry", tags=["Analytics"])
async def analyze_team_chemistry(team_id: int):
    """Analyze team chemistry using mock graph algorithms"""
    
    if team_id not in app.state.db.teams:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Team with ID {team_id} not found"
        )
    
    team = app.state.db.teams[team_id]
    
    # Mock chemistry calculation
    if not team.players:
        chemistry_score = 0.0
    else:
        # Calculate based on average pass accuracy and experience
        avg_accuracy = sum(p.pass_accuracy for p in team.players) / len(team.players)
        avg_minutes = sum(p.minutes_played for p in team.players) / len(team.players)
        chemistry_score = (avg_accuracy / 100 + avg_minutes / 3000) / 2
    
    # Update team chemistry
    app.state.db.teams[team_id].chemistry_score = round(chemistry_score, 3)
    
    return {
        "team_id": team_id,
        "team_name": team.name,
        "chemistry_score": round(chemistry_score, 3),
        "player_count": len(team.players),
        "analysis_method": "pass_accuracy_and_experience",
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Real-time WebSocket connection"""
    
    await app.state.connection_manager.connect(websocket, user_id)
    
    try:
        # Send welcome message
        await websocket.send_text(orjson.dumps({
            "type": "welcome",
            "message": f"Connected to Sports Analytics API",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }).decode())
        
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            try:
                message = orjson.loads(data)
                message_type = message.get("type", "unknown")
                
                if message_type == "ping":
                    await websocket.send_text(orjson.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }).decode())
                
                elif message_type == "subscribe":
                    entity_id = message.get("entity_id")
                    await websocket.send_text(orjson.dumps({
                        "type": "subscription_confirmed",
                        "entity_id": entity_id,
                        "timestamp": datetime.now().isoformat()
                    }).decode())
                
                else:
                    # Echo back unknown messages
                    await websocket.send_text(orjson.dumps({
                        "type": "echo",
                        "original_message": message,
                        "timestamp": datetime.now().isoformat()
                    }).decode())
                    
            except orjson.JSONDecodeError:
                await websocket.send_text(orjson.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                }).decode())
                
    except WebSocketDisconnect:
        app.state.connection_manager.disconnect(websocket, user_id)

@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "api": "operational",
            "database": "operational",
            "ai_predictor": "operational"
        }
    }

@app.get("/stats", tags=["Statistics"])
async def get_api_stats():
    """Get API usage statistics"""
    return {
        "total_players": len(app.state.db.players),
        "total_teams": len(app.state.db.teams),
        "total_predictions": len(app.state.db.predictions),
        "active_websocket_connections": len(app.state.connection_manager.active_connections),
        "cache_size": len(app.state.cache),
        "timestamp": datetime.now().isoformat()
    }

# Background Tasks
async def process_prediction_analytics(prediction: PredictionResponse, request: PredictionRequest):
    """Background task to process prediction analytics"""
    
    # Store analytics data
    analytics_entry = AnalyticsData(
        timestamp=datetime.now(),
        metric_name="prediction_score",
        metric_value=prediction.performance_score,
        player_id=request.player_id
    )
    
    app.state.db.analytics.append(analytics_entry)
    
    # Log for monitoring
    logger.info(f"Processed analytics for prediction {prediction.prediction_id}")

# Exception Handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid value: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting FastAPI Sports Analytics Server...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )