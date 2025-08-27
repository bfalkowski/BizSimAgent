"""Gamification module for BizSimAgent."""

from typing import List, Dict
from .models import Badge, Strategy, SimulationResult, LeaderboardEntry


class GamificationEngine:
    """Engine for calculating badges and scores for strategies."""
    
    def __init__(self):
        self.badges = self._initialize_badges()
    
    def _initialize_badges(self) -> List[Badge]:
        """Initialize all available badges."""
        return [
            # Performance badges
            Badge(
                name="High Flyer",
                description="Exceptional ROI performance",
                category="performance",
                icon="🚀",
                score_bonus=2.0
            ),
            Badge(
                name="Efficiency Master",
                description="Low cost with high returns",
                category="efficiency",
                icon="⚡",
                score_bonus=1.5
            ),
            Badge(
                name="Risk Averse",
                description="Low risk profile",
                category="risk",
                icon="🛡️",
                score_bonus=1.0
            ),
            Badge(
                name="Innovation Leader",
                description="High innovation score",
                category="innovation",
                icon="💡",
                score_bonus=1.5
            ),
            Badge(
                name="Constraint Master",
                description="No constraint violations",
                category="efficiency",
                icon="✅",
                score_bonus=1.0
            ),
            Badge(
                name="Market Dominator",
                description="High market share",
                category="performance",
                icon="👑",
                score_bonus=1.5
            ),
            Badge(
                name="Customer Champion",
                description="High customer acquisition",
                category="performance",
                icon="👥",
                score_bonus=1.0
            ),
            Badge(
                name="Budget Hero",
                description="Under budget execution",
                category="efficiency",
                icon="💰",
                score_bonus=1.0
            ),
            Badge(
                name="Speed Demon",
                description="Fast execution timeline",
                category="efficiency",
                icon="🏃",
                score_bonus=0.5
            ),
            Badge(
                name="Steady Eddie",
                description="Low volatility in outcomes",
                category="risk",
                icon="📊",
                score_bonus=0.5
            )
        ]
    
    def calculate_badges(self, strategy: Strategy, result: SimulationResult) -> List[Badge]:
        """Calculate which badges a strategy has earned."""
        earned_badges = []
        
        for badge in self.badges:
            if self._has_earned_badge(badge, strategy, result):
                earned_badges.append(badge)
        
        return earned_badges
    
    def _has_earned_badge(self, badge: Badge, strategy: Strategy, result: SimulationResult) -> bool:
        """Check if a strategy has earned a specific badge."""
        if badge.name == "High Flyer":
            return result.roi.get('mean', 0) > 0.5  # ROI > 50%
            
        elif badge.name == "Efficiency Master":
            roi = result.roi.get('mean', 0)
            cost = result.total_cost.get('mean', 0)
            return roi > 0.3 and cost < 1000000  # High ROI, low cost
            
        elif badge.name == "Risk Averse":
            return result.risk_score < 3.0  # Low risk score
            
        elif badge.name == "Innovation Leader":
            return strategy.innovation_score >= 8.0  # High innovation score
            
        elif badge.name == "Constraint Master":
            return len(result.constraint_violations) == 0  # No violations
            
        elif badge.name == "Market Dominator":
            market_share = result.metrics.get('market_share', {}).get('mean', 0)
            return market_share > 0.15  # Market share > 15%
            
        elif badge.name == "Customer Champion":
            customers = result.metrics.get('customer_acquisition', {}).get('mean', 0)
            return customers > 10000  # > 10k customers
            
        elif badge.name == "Budget Hero":
            budget_constraint = 1000000  # Example budget constraint
            return result.total_cost.get('mean', 0) < budget_constraint * 0.9  # Under 90% of budget
            
        elif badge.name == "Speed Demon":
            return strategy.expected_timeline_months <= 6  # Fast execution
            
        elif badge.name == "Steady Eddie":
            roi_std = result.roi.get('std', 0)
            roi_mean = result.roi.get('mean', 1)
            return roi_std / (roi_mean + 1e-8) < 0.3  # Low volatility
            
        return False
    
    def calculate_total_score(self, strategy: Strategy, result: SimulationResult, badges: List[Badge]) -> float:
        """Calculate total score for a strategy."""
        base_score = self._calculate_base_score(strategy, result)
        badge_bonus = sum(badge.score_bonus for badge in badges)
        
        return base_score + badge_bonus
    
    def _calculate_base_score(self, strategy: Strategy, result: SimulationResult) -> float:
        """Calculate base score from strategy performance."""
        # ROI component (40% weight)
        roi_score = min(10.0, max(0.0, result.roi.get('mean', 0) * 10))
        
        # Risk component (30% weight) - lower risk = higher score
        risk_score = max(0.0, 10.0 - result.risk_score)
        
        # Efficiency component (20% weight)
        efficiency_score = 0.0
        if result.total_cost.get('mean', 0) > 0:
            roi_per_cost = result.roi.get('mean', 0) / result.total_cost.get('mean', 1)
            efficiency_score = min(10.0, roi_per_cost * 1000000)
        
        # Innovation component (10% weight)
        innovation_score = strategy.innovation_score
        
        # Calculate weighted score
        total_score = (
            roi_score * 0.4 +
            risk_score * 0.3 +
            efficiency_score * 0.2 +
            innovation_score * 0.1
        )
        
        return total_score
    
    def rank_strategies(self, strategies: List[Strategy], results: List[SimulationResult]) -> List[LeaderboardEntry]:
        """Rank strategies based on scores and create leaderboard entries."""
        entries = []
        
        for i, (strategy, result) in enumerate(zip(strategies, results)):
            # Calculate badges and score
            badges = self.calculate_badges(strategy, result)
            total_score = self.calculate_total_score(strategy, result, badges)
            
            # Create leaderboard entry
            entry = LeaderboardEntry(
                rank=0,  # Will be set after sorting
                strategy=strategy,
                simulation_result=result,
                badges=badges,
                total_score=total_score
            )
            entries.append(entry)
        
        # Sort by total score (descending)
        entries.sort(key=lambda x: x.total_score, reverse=True)
        
        # Set ranks
        for i, entry in enumerate(entries):
            entry.rank = i + 1
        
        return entries
    
    def generate_recommendation_notes(self, entry: LeaderboardEntry) -> str:
        """Generate notes explaining why a strategy is recommended."""
        notes = []
        
        # Performance notes
        if entry.simulation_result.roi.get('mean', 0) > 0.3:
            notes.append("Strong ROI performance")
        elif entry.simulation_result.roi.get('mean', 0) > 0.1:
            notes.append("Moderate ROI performance")
        else:
            notes.append("Low ROI performance")
        
        # Risk notes
        if entry.simulation_result.risk_score < 3.0:
            notes.append("Low risk profile")
        elif entry.simulation_result.risk_score < 6.0:
            notes.append("Moderate risk profile")
        else:
            notes.append("High risk profile")
        
        # Badge highlights
        badge_categories = [badge.category for badge in entry.badges]
        if "performance" in badge_categories:
            notes.append("Performance leader")
        if "efficiency" in badge_categories:
            notes.append("Efficiency champion")
        if "innovation" in badge_categories:
            notes.append("Innovation leader")
        
        # Constraint compliance
        if len(entry.simulation_result.constraint_violations) == 0:
            notes.append("All constraints satisfied")
        else:
            notes.append(f"{len(entry.simulation_result.constraint_violations)} constraint violations")
        
        return "; ".join(notes)
