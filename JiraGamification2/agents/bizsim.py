"""WaaP agent wrapper for BizSimAgent."""

import json
import sys
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path to import bizsim modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from bizsim.models import AskSpec, Strategy, Leaderboard, Recommendation
from bizsim.simulate import MonteCarloSimulator
from bizsim.gamify import GamificationEngine
from bizsim.cli import _create_stub_spec, _generate_stub_strategies, _create_recommendation

console = Console()


def main():
    """Main entry point for the WaaP agent."""
    console.print(Panel.fit("🤖 [bold blue]BizSimAgent - WaaP Mode[/bold blue]"))
    
    # Check if context.json exists
    context_path = Path("context.json")
    if not context_path.exists():
        console.print("[red]❌ context.json not found. Please ensure it exists in the current directory.[/red]")
        sys.exit(1)
    
    # Load context
    try:
        with open(context_path, 'r') as f:
            context = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]❌ Error parsing context.json: {e}[/red]")
        sys.exit(1)
    
    # Validate required inputs
    required_inputs = ['ask.text', 'constraints.json', 'priors.json']
    missing_inputs = [input_name for input_name in required_inputs if input_name not in context]
    
    if missing_inputs:
        console.print(f"[red]❌ Missing required inputs: {', '.join(missing_inputs)}[/red]")
        sys.exit(1)
    
    console.print("📋 [green]Context loaded successfully[/green]")
    
    # Create output directories
    results_dir = Path("results")
    artifacts_dir = Path("artifacts/sim")
    results_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Step 1: Translate business ask
        task1 = progress.add_task("🔍 Translating business ask...", total=None)
        try:
            business_ask = context['ask.text']
            spec = _create_stub_spec(business_ask)
            progress.update(task1, description="✅ Business ask translated")
        except Exception as e:
            console.print(f"[red]❌ Error in translation step: {e}[/red]")
            sys.exit(1)
        
        # Step 2: Generate strategies
        task2 = progress.add_task("🚀 Generating strategies...", total=None)
        try:
            strategies = _generate_stub_strategies(spec)
            progress.update(task2, description="✅ Strategies generated")
        except Exception as e:
            console.print(f"[red]❌ Error in strategy generation: {e}[/red]")
            sys.exit(1)
        
        # Step 3: Run simulations
        task3 = progress.add_task("🎲 Running Monte Carlo simulations...", total=None)
        try:
            simulator = MonteCarloSimulator(spec)
            gamification = GamificationEngine()
            
            # Run simulations with default trial count
            trials = 1000
            results = []
            for strategy in strategies:
                result = simulator.simulate_strategy(strategy, trials)
                results.append(result)
            
            progress.update(task3, description="✅ Simulations completed")
        except Exception as e:
            console.print(f"[red]❌ Error in simulation step: {e}[/red]")
            sys.exit(1)
        
        # Step 4: Create leaderboard
        task4 = progress.add_task("🏆 Creating leaderboard...", total=None)
        try:
            leaderboard_entries = gamification.rank_strategies(strategies, results)
            
            # Add recommendation notes
            for entry in leaderboard_entries:
                entry.recommendation_notes = gamification.generate_recommendation_notes(entry)
            
            leaderboard = Leaderboard(
                business_ask=spec.business_ask,
                simulation_parameters={
                    "trials": trials,
                    "distribution_types": ", ".join(set(p.type.value for p in spec.priors.values())),
                    "levers_count": len(spec.levers),
                    "constraints_count": len(spec.constraints)
                },
                strategies=leaderboard_entries,
                generated_at=datetime.now().isoformat(),
                total_trials=trials * len(strategies)
            )
            
            progress.update(task4, description="✅ Leaderboard created")
        except Exception as e:
            console.print(f"[red]❌ Error in leaderboard creation: {e}[/red]")
            sys.exit(1)
        
        # Step 5: Generate recommendation
        task5 = progress.add_task("🎯 Generating recommendation...", total=None)
        try:
            top_entry = leaderboard_entries[0]
            recommendation = _create_recommendation(top_entry, leaderboard)
            progress.update(task5, description="✅ Recommendation generated")
        except Exception as e:
            console.print(f"[red]❌ Error in recommendation generation: {e}[/red]")
            sys.exit(1)
    
    # Save outputs
    try:
        # Save leaderboard
        leaderboard_path = results_dir / "sim.leaderboard.json"
        with open(leaderboard_path, 'w') as f:
            json.dump(leaderboard.model_dump(), f, indent=2)
        
        # Save recommendation
        recommendation_path = results_dir / "sim.recommendation.json"
        with open(recommendation_path, 'w') as f:
            json.dump(recommendation.model_dump(), f, indent=2)
        
        # Generate and save charts
        _generate_charts(leaderboard, artifacts_dir)
        
    except Exception as e:
        console.print(f"[red]❌ Error saving outputs: {e}[/red]")
        sys.exit(1)
    
    # Display results
    console.print("\n" + "="*60)
    console.print(Panel.fit("🏆 [bold green]SIMULATION COMPLETE[/bold green]"))
    
    # Show top strategy
    top_strategy = leaderboard.strategies[0]
    console.print(f"🥇 **Top Strategy**: {top_strategy.strategy.name}")
    console.print(f"📊 **Score**: {top_strategy.total_score:.2f}")
    console.print(f"💰 **Expected ROI**: {top_strategy.simulation_result.roi.get('mean', 0):.1%}")
    console.print(f"⚠️  **Risk Score**: {top_strategy.simulation_result.risk_score:.1f}")
    console.print(f"🎖️  **Badges Earned**: {len(top_strategy.badges)}")
    
    # Show recommendation
    console.print(f"\n💡 **Recommendation**: {recommendation.rationale}")
    console.print(f"🎯 **Confidence**: {recommendation.confidence_score:.1%}")
    
    # Show key benefits and risks
    if recommendation.key_benefits:
        console.print(f"\n✅ **Key Benefits**: {', '.join(recommendation.key_benefits)}")
    if recommendation.key_risks:
        console.print(f"⚠️  **Key Risks**: {', '.join(recommendation.key_risks)}")
    
    # Show outputs
    console.print(f"\n📁 **Outputs**:")
    console.print(f"   📊 Leaderboard: [green]{leaderboard_path}[/green]")
    console.print(f"   🎯 Recommendation: [green]{recommendation_path}[/green]")
    console.print(f"   📈 Charts: [green]{artifacts_dir}[/green]")
    
    console.print("\n" + "="*60)
    console.print("[green]✅ BizSimAgent completed successfully![/green]")


def _generate_charts(leaderboard: Leaderboard, artifacts_dir: Path):
    """Generate and save visualization charts."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'BizSim Results: {leaderboard.business_ask[:50]}...', fontsize=16, fontweight='bold')
        
        # 1. Score vs Risk scatter plot
        scores = [entry.total_score for entry in leaderboard.strategies]
        risks = [entry.simulation_result.risk_score for entry in leaderboard.strategies]
        names = [entry.strategy.name for entry in leaderboard.strategies]
        
        ax1 = axes[0, 0]
        scatter = ax1.scatter(risks, scores, s=100, alpha=0.7, c=scores, cmap='viridis')
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Total Score')
        ax1.set_title('Score vs Risk Profile')
        
        # Add strategy names as annotations
        for i, name in enumerate(names):
            ax1.annotate(name[:15] + '...' if len(name) > 15 else name, 
                         (risks[i], scores[i]), 
                         xytext=(5, 5), textcoords='offset points', 
                         fontsize=8, alpha=0.8)
        
        # 2. ROI distribution
        ax2 = axes[0, 1]
        roi_means = [entry.simulation_result.roi.get('mean', 0) for entry in leaderboard.strategies]
        roi_stds = [entry.simulation_result.roi.get('std', 0) for entry in leaderboard.strategies]
        
        bars = ax2.bar(range(len(roi_means)), roi_means, yerr=roi_stds, 
                       capsize=5, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Strategy Index')
        ax2.set_ylabel('ROI')
        ax2.set_title('ROI by Strategy (with std dev)')
        ax2.set_xticks(range(len(roi_means)))
        ax2.set_xticklabels([f'S{i+1}' for i in range(len(roi_means))])
        
        # 3. Badge count by strategy
        ax3 = axes[1, 0]
        badge_counts = [len(entry.badges) for entry in leaderboard.strategies]
        badge_categories = {}
        
        for entry in leaderboard.strategies:
            for badge in entry.badges:
                category = badge.category
                if category not in badge_categories:
                    badge_categories[category] = [0] * len(leaderboard.strategies)
                badge_categories[category][entry.rank - 1] += 1
        
        if badge_categories:
            x = range(len(leaderboard.strategies))
            bottom = [0] * len(leaderboard.strategies)
            
            for category, counts in badge_categories.items():
                ax3.bar(x, counts, bottom=bottom, label=category, alpha=0.8)
                bottom = [bottom[i] + counts[i] for i in range(len(bottom))]
            
            ax3.set_xlabel('Strategy Rank')
            ax3.set_ylabel('Number of Badges')
            ax3.set_title('Badges by Category and Strategy')
            ax3.set_xticks(x)
            ax3.set_xticklabels([f'#{i+1}' for i in x])
            ax3.legend()
        
        # 4. Cost vs ROI efficiency
        ax4 = axes[1, 1]
        costs = [entry.simulation_result.total_cost.get('mean', 0) for entry in leaderboard.strategies]
        
        scatter2 = ax4.scatter(costs, roi_means, s=100, alpha=0.7, c=scores, cmap='plasma')
        ax4.set_xlabel('Expected Cost (USD)')
        ax4.set_ylabel('Expected ROI')
        ax4.set_title('Cost vs ROI Efficiency')
        
        # Add colorbar
        cbar = plt.colorbar(scatter2, ax=ax4)
        cbar.set_label('Total Score')
        
        # Adjust layout and save
        plt.tight_layout()
        chart_path = artifacts_dir / "simulation_results.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"📈 Charts saved to [green]{chart_path}[/green]")
        
    except ImportError:
        console.print("[yellow]⚠️  matplotlib/seaborn not available. Skipping chart generation.[/yellow]")
    except Exception as e:
        console.print(f"[yellow]⚠️  Chart generation failed: {e}[/yellow]")


if __name__ == "__main__":
    main()
