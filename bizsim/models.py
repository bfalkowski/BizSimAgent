"""Pydantic data models for BizSimAgent."""

from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
import numpy as np


class DistributionType(str, Enum):
    """Types of probability distributions for simulation priors."""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    BETA = "beta"
    TRIANGULAR = "triangular"


class PriorDistribution(BaseModel):
    """Probability distribution for a simulation parameter."""
    type: DistributionType
    parameters: Dict[str, float] = Field(description="Distribution parameters")
    
    def sample(self, size: int = 1) -> np.ndarray:
        """Generate random samples from the distribution."""
        if self.type == DistributionType.NORMAL:
            return np.random.normal(
                self.parameters["mean"], 
                self.parameters["std"], 
                size
            )
        elif self.type == DistributionType.LOGNORMAL:
            return np.random.lognormal(
                self.parameters["mean"], 
                self.parameters["std"], 
                size
            )
        elif self.type == DistributionType.BETA:
            return np.random.beta(
                self.parameters["alpha"], 
                self.parameters["beta"], 
                size
            )
        elif self.type == DistributionType.TRIANGULAR:
            return np.random.triangular(
                self.parameters["left"], 
                self.parameters["mode"], 
                self.parameters["right"], 
                size
            )
        else:
            raise ValueError(f"Unknown distribution type: {self.type}")


class Lever(BaseModel):
    """A decision lever that can be adjusted in a strategy."""
    name: str
    description: str
    min_value: float
    max_value: float
    default_value: float
    unit: str
    cost_per_unit: Optional[float] = None
    risk_factor: float = Field(default=1.0, ge=0.0, le=10.0)


class Constraint(BaseModel):
    """A business constraint that must be satisfied."""
    name: str
    description: str
    constraint_type: str  # "budget", "timeline", "resource", "regulatory"
    value: float
    operator: str  # "le", "ge", "eq"
    priority: int = Field(default=1, ge=1, le=5)


class Metric(BaseModel):
    """A business metric to be optimized or tracked."""
    name: str
    description: str
    target: Optional[float] = None
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    is_higher_better: bool = True


class AskSpec(BaseModel):
    """Complete specification extracted from a business ask."""
    business_ask: str
    levers: List[Lever]
    constraints: List[Constraint]
    metrics: List[Metric]
    priors: Dict[str, PriorDistribution]
    timeline_months: int = 12
    budget_cap: Optional[float] = None


class Strategy(BaseModel):
    """A candidate strategy with specific lever values."""
    name: str
    description: str
    lever_values: Dict[str, float]
    expected_cost: float
    expected_timeline_months: int
    risk_profile: str = "medium"  # "low", "medium", "high"
    innovation_score: float = Field(default=5.0, ge=1.0, le=10.0)


class SimulationResult(BaseModel):
    """Results from a single strategy simulation."""
    strategy_name: str
    trials: int
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {mean, std, p10, p90, cvar}
    total_cost: Dict[str, float]  # {mean, std, p10, p90, cvar}
    roi: Dict[str, float]  # {mean, std, p10, p90, cvar}
    risk_score: float
    constraint_violations: List[str]
    success_rate: float


class Badge(BaseModel):
    """A gamification badge for strategy performance."""
    name: str
    description: str
    category: str  # "performance", "risk", "innovation", "efficiency"
    icon: str
    score_bonus: float


class LeaderboardEntry(BaseModel):
    """A single entry in the strategy leaderboard."""
    rank: int
    strategy: Strategy
    simulation_result: SimulationResult
    badges: List[Badge]
    total_score: float
    recommendation_notes: Optional[str] = None


class Leaderboard(BaseModel):
    """Complete leaderboard of all simulated strategies."""
    business_ask: str
    simulation_parameters: Dict[str, Union[int, str]]
    strategies: List[LeaderboardEntry]
    generated_at: str
    total_trials: int


class Recommendation(BaseModel):
    """Final recommendation with rationale."""
    recommended_strategy: Strategy
    rationale: str
    key_benefits: List[str]
    key_risks: List[str]
    next_steps: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
