from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Player(BaseModel):
    id: int
    name: str
    position: str
    team: str
    goals: int
    assists: int
    passes: int
    pass_accuracy: float
    minutes_played: int
    yellow_cards: int
    red_cards: int

class Team(BaseModel):
    id: int
    name: str
    league: str
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    points: int
    matches_played: int

class PredictionRequest(BaseModel):
    goals: int
    assists: int
    passes: int
    pass_accuracy: float
    minutes_played: int
    yellow_cards: int = 0
    red_cards: int = 0

class PredictionResponse(BaseModel):
    performance_score: float
    win_probability: float
    rating: str
    insights: List[str]