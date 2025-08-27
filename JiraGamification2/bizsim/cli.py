"""CLI interface for BizSimAgent."""

import json
import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from datetime import datetime

from .models import (
    AskSpec, Strategy, Leaderboard, Recommendation,
    PriorDistribution, DistributionType, Lever, Constraint, Metric
)
from .simulate import MonteCarloSimulator
from .gamify import GamificationEngine

app = typer.Typer(help="BizSimAgent - Business Simulation and Gamification Tool")
console = Console()


@app.command()
def translate(
    ask: str = typer.Option(..., "--ask", "-a", help="Business ask text or path to file"),
    out: str = typer.Option(..., "--out", "-o", help="Output JSON file path")
):
    """Translate a business ask into a structured specification."""
    console.print(Panel.fit("🔍 [bold blue]Translating Business Ask[/bold blue]"))
    
    # Read business ask
    if Path(ask).exists():
        with open(ask, 'r') as f:
            business_ask = f.read().strip()
    else:
        business_ask = ask
    
    console.print(f"📝 Business Ask: {business_ask}")
    
    # Create stub specification (in real implementation, this would call Q/Bedrock)
    spec = _create_stub_spec(business_ask)
    
    # Save to output file
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(spec.model_dump(), f, indent=2)
    
    console.print(f"✅ Specification saved to [green]{output_path}[/green]")
    console.print(f"📊 Generated {len(spec.levers)} levers, {len(spec.constraints)} constraints, {len(spec.metrics)} metrics")


@app.command()
def generate(
    spec: str = typer.Option(..., "--spec", "-s", help="Input specification JSON file"),
    out: str = typer.Option(..., "--out", "-o", help="Output strategies JSON file")
):
    """Generate candidate strategies from a specification."""
    console.print(Panel.fit("🚀 [bold green]Generating Strategies[/bold green]"))
    
    # Load specification
    with open(spec, 'r') as f:
        spec_data = json.load(f)
    ask_spec = AskSpec(**spec_data)
    
    console.print(f"📋 Loaded specification with {len(ask_spec.levers)} levers")
    
    # Generate strategies (stub implementation)
    strategies = _generate_stub_strategies(ask_spec)
    
    # Save strategies
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump([s.model_dump() for s in strategies], f, indent=2)
    
    console.print(f"✅ Generated {len(strategies)} strategies")
    console.print(f"💾 Strategies saved to [green]{output_path}[/green]")


@app.command()
def simulate(
    spec: str = typer.Option(..., "--spec", "-s", help="Input specification JSON file"),
    strategies: str = typer.Option(..., "--strategies", "-st", help="Input strategies JSON file"),
    out: str = typer.Option(..., "--out", "-o", help="Output leaderboard JSON file"),
    trials: int = typer.Option(5000, "--trials", "-t", help="Number of Monte Carlo trials")
):
    """Run Monte Carlo simulations for all strategies."""
    console.print(Panel.fit("🎲 [bold purple]Running Monte Carlo Simulations[/bold purple]"))
    
    # Load specification and strategies
    with open(spec, 'r') as f:
        spec_data = json.load(f)
    ask_spec = AskSpec(**spec_data)
    
    with open(strategies, 'r') as f:
        strategies_data = json.load(f)
    strategy_list = [Strategy(**s) for s in strategies_data]
    
    console.print(f"📊 Running {trials} trials for {len(strategy_list)} strategies")
    
    # Initialize simulator and gamification engine
    simulator = MonteCarloSimulator(ask_spec)
    gamification = GamificationEngine()
    
    # Run simulations
    results = []
    for strategy in track(strategy_list, description="Simulating strategies..."):
        result = simulator.simulate_strategy(strategy, trials)
        results.append(result)
    
    # Create leaderboard
    leaderboard_entries = gamification.rank_strategies(strategy_list, results)
    
    # Add recommendation notes
    for entry in leaderboard_entries:
        entry.recommendation_notes = gamification.generate_recommendation_notes(entry)
    
            # Create leaderboard
        leaderboard = Leaderboard(
            business_ask=ask_spec.business_ask,
            simulation_parameters={
                "trials": trials,
                "distribution_types": ", ".join(set(p.type.value for p in ask_spec.priors.values())),
                "levers_count": len(ask_spec.levers),
                "constraints_count": len(ask_spec.constraints)
            },
            strategies=leaderboard_entries,
            generated_at=datetime.now().isoformat(),
            total_trials=trials * len(strategy_list)
        )
    
    # Save leaderboard
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(leaderboard.model_dump(), f, indent=2)
    
    # Display results
    _display_leaderboard(leaderboard)
    
    console.print(f"✅ Leaderboard saved to [green]{output_path}[/green]")


@app.command()
def recommend(
    leaderboard: str = typer.Option(..., "--leaderboard", "-l", help="Input leaderboard JSON file"),
    out: str = typer.Option(..., "--out", "-o", help="Output recommendation JSON file")
):
    """Generate a recommendation from the leaderboard."""
    console.print(Panel.fit("🎯 [bold red]Generating Recommendation[/bold red]"))
    
    # Load leaderboard
    with open(leaderboard, 'r') as f:
        leaderboard_data = json.load(f)
    leaderboard_obj = Leaderboard(**leaderboard_data)
    
    # Get top strategy
    top_strategy = leaderboard_obj.strategies[0]
    
    console.print(f"🥇 Top Strategy: [bold]{top_strategy.strategy.name}[/bold]")
    console.print(f"📊 Score: [green]{top_strategy.total_score:.2f}[/green]")
    console.print(f"🎖️  Badges: {len(top_strategy.badges)}")
    
    # Create recommendation
    recommendation = _create_recommendation(top_strategy, leaderboard_obj)
    
    # Save recommendation
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(recommendation.model_dump(), f, indent=2)
    
    console.print(f"✅ Recommendation saved to [green]{output_path}[/green]")


def _create_stub_spec(business_ask: str) -> AskSpec:
    """Create a stub specification with example levers, constraints, and metrics."""
    # Example levers
    levers = [
        Lever(
            name="marketing_budget",
            description="Marketing and advertising budget",
            min_value=50000,
            max_value=500000,
            default_value=200000,
            unit="USD",
            cost_per_unit=1.0,
            risk_factor=2.0
        ),
        Lever(
            name="team_size",
            description="Development team size",
            min_value=2,
            max_value=20,
            default_value=8,
            unit="people",
            cost_per_unit=150000,
            risk_factor=1.5
        ),
        Lever(
            name="timeline",
            description="Project timeline in months",
            min_value=3,
            max_value=24,
            default_value=12,
            unit="months",
            cost_per_unit=50000,
            risk_factor=1.0
        ),
        Lever(
            name="feature_scope",
            description="Number of features to implement",
            min_value=5,
            max_value=50,
            default_value=20,
            unit="features",
            cost_per_unit=10000,
            risk_factor=2.5
        )
    ]
    
    # Example constraints
    constraints = [
        Constraint(
            name="budget_limit",
            description="Maximum total budget",
            constraint_type="budget",
            value=1000000,
            operator="le",
            priority=1
        ),
        Constraint(
            name="timeline_limit",
            description="Maximum project timeline",
            constraint_type="timeline",
            value=18,
            operator="le",
            priority=2
        )
    ]
    
    # Example metrics
    metrics = [
        Metric(
            name="revenue",
            description="Expected annual revenue",
            target=500000,
            weight=8.0,
            is_higher_better=True
        ),
        Metric(
            name="customer_acquisition",
            description="Number of customers acquired",
            target=1000,
            weight=6.0,
            is_higher_better=True
        ),
        Metric(
            name="market_share",
            description="Market share percentage",
            target=0.1,
            weight=7.0,
            is_higher_better=True
        )
    ]
    
    # Example priors
    priors = {
        "market_size": PriorDistribution(
            type=DistributionType.LOGNORMAL,
            parameters={"mean": 10.0, "std": 0.5}
        ),
        "conversion_rate": PriorDistribution(
            type=DistributionType.BETA,
            parameters={"alpha": 2.0, "beta": 20.0}
        ),
        "price": PriorDistribution(
            type=DistributionType.NORMAL,
            parameters={"mean": 100.0, "std": 20.0}
        ),
        "acquisition_cost": PriorDistribution(
            type=DistributionType.TRIANGULAR,
            parameters={"left": 30.0, "mode": 50.0, "right": 80.0}
        ),
        "competitive_advantage": PriorDistribution(
            type=DistributionType.BETA,
            parameters={"alpha": 3.0, "beta": 7.0}
        ),
        "cost_variance": PriorDistribution(
            type=DistributionType.NORMAL,
            parameters={"mean": 0.0, "std": 0.2}
        )
    }
    
    return AskSpec(
        business_ask=business_ask,
        levers=levers,
        constraints=constraints,
        metrics=metrics,
        priors=priors,
        timeline_months=12,
        budget_cap=1000000
    )


def _generate_stub_strategies(spec: AskSpec) -> List[Strategy]:
    """Generate stub strategies by sampling across lever ranges."""
    strategies = []
    
    # Conservative strategy
    strategies.append(Strategy(
        name="Conservative Approach",
        description="Low-risk, steady growth strategy",
        lever_values={
            "marketing_budget": 100000,
            "team_size": 5,
            "timeline": 15,
            "feature_scope": 15
        },
        expected_cost=600000,
        expected_timeline_months=15,
        risk_profile="low",
        innovation_score=4.0
    ))
    
    # Balanced strategy
    strategies.append(Strategy(
        name="Balanced Growth",
        description="Moderate risk and investment strategy",
        lever_values={
            "marketing_budget": 250000,
            "team_size": 10,
            "timeline": 12,
            "feature_scope": 25
        },
        expected_cost=800000,
        expected_timeline_months=12,
        risk_profile="medium",
        innovation_score=6.0
    ))
    
    # Aggressive strategy
    strategies.append(Strategy(
        name="Aggressive Expansion",
        description="High-risk, high-reward strategy",
        lever_values={
            "marketing_budget": 400000,
            "team_size": 15,
            "timeline": 8,
            "feature_scope": 40
        },
        expected_cost=1200000,
        expected_timeline_months=8,
        risk_profile="high",
        innovation_score=8.5
    ))
    
    # Innovation-focused strategy
    strategies.append(Strategy(
        name="Innovation Leader",
        description="Focus on cutting-edge features and rapid development",
        lever_values={
            "marketing_budget": 300000,
            "team_size": 12,
            "timeline": 10,
            "feature_scope": 35
        },
        expected_cost=900000,
        expected_timeline_months=10,
        risk_profile="medium",
        innovation_score=9.0
    ))
    
    # Cost-effective strategy
    strategies.append(Strategy(
        name="Cost Optimizer",
        description="Minimize costs while maintaining quality",
        lever_values={
            "marketing_budget": 75000,
            "team_size": 6,
            "timeline": 18,
            "feature_scope": 18
        },
        expected_cost=500000,
        expected_timeline_months=18,
        risk_profile="low",
        innovation_score=5.5
    ))
    
    return strategies


def _create_recommendation(top_entry, leaderboard: Leaderboard) -> Recommendation:
    """Create a recommendation from the top strategy."""
    strategy = top_entry.strategy
    result = top_entry.simulation_result
    
    # Generate rationale
    rationale_parts = []
    
    if result.roi.get('mean', 0) > 0.3:
        rationale_parts.append("demonstrates exceptional ROI potential")
    elif result.roi.get('mean', 0) > 0.1:
        rationale_parts.append("shows strong return on investment")
    else:
        rationale_parts.append("offers reasonable returns")
    
    if result.risk_score < 3.0:
        rationale_parts.append("with a low-risk profile")
    elif result.risk_score < 6.0:
        rationale_parts.append("with moderate risk")
    else:
        rationale_parts.append("with higher risk but high potential")
    
    if len(top_entry.badges) > 3:
        rationale_parts.append("and has earned multiple performance badges")
    
    rationale = f"This strategy {' '.join(rationale_parts)}."
    
    # Key benefits
    key_benefits = []
    if result.roi.get('mean', 0) > 0.2:
        key_benefits.append("High expected ROI")
    if result.risk_score < 4.0:
        key_benefits.append("Low risk profile")
    if len(result.constraint_violations) == 0:
        key_benefits.append("All constraints satisfied")
    if strategy.innovation_score > 7.0:
        key_benefits.append("High innovation potential")
    
    # Key risks
    key_risks = []
    if result.risk_score > 6.0:
        key_risks.append("Higher risk profile")
    if result.total_cost.get('mean', 0) > 800000:
        key_risks.append("High investment requirement")
    if strategy.expected_timeline_months > 15:
        key_risks.append("Extended timeline")
    
    # Next steps
    next_steps = [
        "Conduct detailed feasibility analysis",
        "Develop implementation roadmap",
        "Secure stakeholder approval",
        "Begin pilot phase"
    ]
    
    # Confidence score based on simulation results
    confidence_score = min(0.95, 0.5 + (top_entry.total_score / 20.0))
    
    return Recommendation(
        recommended_strategy=strategy,
        rationale=rationale,
        key_benefits=key_benefits,
        key_risks=key_risks,
        next_steps=next_steps,
        confidence_score=confidence_score
    )


def _display_leaderboard(leaderboard: Leaderboard):
    """Display the leaderboard in a rich table."""
    table = Table(title="🏆 Strategy Leaderboard")
    
    table.add_column("Rank", style="cyan", no_wrap=True)
    table.add_column("Strategy", style="magenta")
    table.add_column("Score", style="green")
    table.add_column("ROI", style="yellow")
    table.add_column("Risk", style="red")
    table.add_column("Badges", style="blue")
    
    for entry in leaderboard.strategies[:10]:  # Top 10
        badges_str = " ".join([badge.icon for badge in entry.badges[:3]])
        if len(entry.badges) > 3:
            badges_str += f" (+{len(entry.badges) - 3})"
        
        table.add_row(
            str(entry.rank),
            entry.strategy.name[:30] + "..." if len(entry.strategy.name) > 30 else entry.strategy.name,
            f"{entry.total_score:.2f}",
            f"{entry.simulation_result.roi.get('mean', 0):.1%}",
            f"{entry.simulation_result.risk_score:.1f}",
            badges_str
        )
    
    console.print(table)


if __name__ == "__main__":
    app()
