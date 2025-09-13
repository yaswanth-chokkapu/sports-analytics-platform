from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import Player, Team, PredictionRequest, PredictionResponse
from typing import List
import math

app = FastAPI(title="Sports Performance Analytics API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample data
SAMPLE_PLAYERS = [
    Player(
        id=1, name="Lionel Messi", position="Forward", team="Inter Miami",
        goals=15, assists=12, passes=2450, pass_accuracy=89.5,
        minutes_played=2340, yellow_cards=2, red_cards=0
    ),
    Player(
        id=2, name="Kevin De Bruyne", position="Midfielder", team="Man City",
        goals=8, assists=18, passes=3200, pass_accuracy=91.2,
        minutes_played=2800, yellow_cards=3, red_cards=0
    ),
    Player(
        id=3, name="Virgil van Dijk", position="Defender", team="Liverpool",
        goals=3, assists=2, passes=2900, pass_accuracy=93.8,
        minutes_played=3100, yellow_cards=4, red_cards=1
    ),
    Player(
        id=4, name="Kylian Mbappé", position="Forward", team="PSG",
        goals=22, assists=8, passes=1850, pass_accuracy=84.3,
        minutes_played=2650, yellow_cards=1, red_cards=0
    ),
    Player(
        id=5, name="Luka Modrić", position="Midfielder", team="Real Madrid",
        goals=4, assists=10, passes=3400, pass_accuracy=92.1,
        minutes_played=2950, yellow_cards=5, red_cards=0
    )
]

SAMPLE_TEAMS = [
    Team(
        id=1, name="Manchester City", league="Premier League",
        wins=25, draws=7, losses=6, goals_for=89, goals_against=31,
        points=82, matches_played=38
    ),
    Team(
        id=2, name="Arsenal", league="Premier League",
        wins=24, draws=6, losses=8, goals_for=88, goals_against=43,
        points=78, matches_played=38
    ),
    Team(
        id=3, name="Real Madrid", league="La Liga",
        wins=28, draws=8, losses=2, goals_for=75, goals_against=26,
        points=92, matches_played=38
    ),
    Team(
        id=4, name="Barcelona", league="La Liga",
        wins=24, draws=10, losses=4, goals_for=70, goals_against=35,
        points=82, matches_played=38
    ),
    Team(
        id=5, name="Bayern Munich", league="Bundesliga",
        wins=26, draws=7, losses=5, goals_for=92, goals_against=38,
        points=85, matches_played=38
    )
]

@app.get("/")
async def root():
    return {"message": "Sports Performance Analytics API", "version": "1.0.0"}

@app.get("/players", response_model=List[Player])
async def get_players():
    """Get all players with their statistics"""
    return SAMPLE_PLAYERS

@app.get("/players/{player_id}", response_model=Player)
async def get_player(player_id: int):
    """Get specific player by ID"""
    player = next((p for p in SAMPLE_PLAYERS if p.id == player_id), None)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player

@app.get("/teams", response_model=List[Team])
async def get_teams():
    """Get all teams with their statistics"""
    return SAMPLE_TEAMS

@app.get("/teams/{team_id}", response_model=Team)
async def get_team(team_id: int):
    """Get specific team by ID"""
    team = next((t for t in SAMPLE_TEAMS if t.id == team_id), None)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team

@app.post("/predict", response_model=PredictionResponse)
async def predict_performance(request: PredictionRequest):
    """Predict player performance and win probability"""
    
    # Simple performance calculation algorithm
    goals_weight = 0.3
    assists_weight = 0.25
    pass_accuracy_weight = 0.2
    passes_weight = 0.15
    discipline_weight = 0.1
    
    # Normalize values
    goals_score = min(request.goals / 30 * 100, 100)  # Max 30 goals = 100 points
    assists_score = min(request.assists / 20 * 100, 100)  # Max 20 assists = 100 points
    pass_accuracy_score = request.pass_accuracy
    passes_score = min(request.passes / 4000 * 100, 100)  # Max 4000 passes = 100 points
    discipline_score = max(100 - (request.yellow_cards * 5 + request.red_cards * 20), 0)
    
    # Calculate performance score
    performance_score = (
        goals_score * goals_weight +
        assists_score * assists_weight +
        pass_accuracy_score * pass_accuracy_weight +
        passes_score * passes_weight +
        discipline_score * discipline_weight
    )
    
    # Calculate win probability (based on performance score with some randomness)
    base_win_prob = performance_score / 100 * 0.8  # Max 80% from performance
    win_probability = min(base_win_prob + 0.1, 0.95) * 100  # Add base 10%, cap at 95%
    
    # Determine rating
    if performance_score >= 90:
        rating = "World Class"
    elif performance_score >= 80:
        rating = "Excellent"
    elif performance_score >= 70:
        rating = "Very Good"
    elif performance_score >= 60:
        rating = "Good"
    elif performance_score >= 50:
        rating = "Average"
    else:
        rating = "Below Average"
    
    # Generate insights
    insights = []
    if request.goals > 15:
        insights.append("🔥 Exceptional goal-scoring ability")
    if request.assists > 10:
        insights.append("🎯 Great playmaking skills")
    if request.pass_accuracy > 90:
        insights.append("⚡ Outstanding passing accuracy")
    if request.yellow_cards == 0 and request.red_cards == 0:
        insights.append("🏆 Perfect discipline record")
    if performance_score > 85:
        insights.append("⭐ Elite performance level")
    
    if not insights:
        insights.append("📈 Room for improvement in key areas")
    
    return PredictionResponse(
        performance_score=round(performance_score, 2),
        win_probability=round(win_probability, 2),
        rating=rating,
        insights=insights
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "sports-analytics-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)