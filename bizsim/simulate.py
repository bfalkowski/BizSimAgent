"""Monte Carlo simulation engine for BizSimAgent."""

import numpy as np
from typing import Dict, List, Tuple
from .models import (
    AskSpec, Strategy, SimulationResult, PriorDistribution, 
    DistributionType, Constraint
)


class MonteCarloSimulator:
    """Monte Carlo simulation engine for business strategies."""
    
    def __init__(self, spec: AskSpec):
        self.spec = spec
        self.priors = spec.priors
        
    def simulate_strategy(self, strategy: Strategy, trials: int = 1000) -> SimulationResult:
        """Run Monte Carlo simulation for a single strategy."""
        # Generate samples for all prior distributions
        samples = self._generate_samples(trials)
        
        # Calculate metrics for each trial
        trial_results = []
        for i in range(trials):
            trial_result = self._evaluate_trial(strategy, samples, i)
            trial_results.append(trial_result)
        
        # Aggregate results
        return self._aggregate_results(strategy, trial_results, trials)
    
    def _generate_samples(self, trials: int) -> Dict[str, np.ndarray]:
        """Generate samples from all prior distributions."""
        samples = {}
        for param_name, prior in self.priors.items():
            samples[param_name] = prior.sample(trials)
        return samples
    
    def _evaluate_trial(self, strategy: Strategy, samples: Dict[str, np.ndarray], trial_idx: int) -> Dict:
        """Evaluate a single simulation trial."""
        # Extract values for this trial
        trial_values = {param: samples[param][trial_idx] for param in samples}
        
        # Calculate metrics based on lever values and trial parameters
        metrics = self._calculate_metrics(strategy, trial_values)
        
        # Calculate costs
        total_cost = self._calculate_cost(strategy, trial_values)
        
        # Calculate ROI (simplified)
        roi = self._calculate_roi(metrics, total_cost)
        
        # Check constraints
        constraint_violations = self._check_constraints(strategy, trial_values, total_cost)
        
        return {
            'metrics': metrics,
            'total_cost': total_cost,
            'roi': roi,
            'constraint_violations': constraint_violations
        }
    
    def _calculate_metrics(self, strategy: Strategy, trial_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate business metrics for a trial."""
        metrics = {}
        
        # Example metric calculations (these would be more sophisticated in practice)
        for metric in self.spec.metrics:
            if metric.name == "revenue":
                # Revenue based on market size and conversion rate
                market_size = trial_values.get('market_size', 1000000)
                conversion_rate = trial_values.get('conversion_rate', 0.05)
                price = trial_values.get('price', 100)
                metrics[metric.name] = market_size * conversion_rate * price
                
            elif metric.name == "customer_acquisition":
                # Customer acquisition based on marketing spend and efficiency
                marketing_spend = trial_values.get('marketing_spend', 100000)
                acquisition_cost = trial_values.get('acquisition_cost', 50)
                metrics[metric.name] = marketing_spend / acquisition_cost
                
            elif metric.name == "market_share":
                # Market share based on competitive position
                competitive_advantage = trial_values.get('competitive_advantage', 0.1)
                base_market_share = 0.05
                metrics[metric.name] = base_market_share * (1 + competitive_advantage)
                
            else:
                # Default metric calculation
                metrics[metric.name] = trial_values.get(metric.name, 0.0)
        
        return metrics
    
    def _calculate_cost(self, strategy: Strategy, trial_values: Dict[str, float]) -> float:
        """Calculate total cost for a trial."""
        base_cost = strategy.expected_cost
        
        # Add variability based on trial parameters
        cost_multiplier = 1.0
        if 'cost_variance' in trial_values:
            cost_multiplier = 1.0 + trial_values['cost_variance']
        
        # Add lever-specific costs
        lever_costs = 0.0
        for lever_name, lever_value in strategy.lever_values.items():
            if lever_name in self.spec.levers:
                lever = next(l for l in self.spec.levers if l.name == lever_name)
                if lever.cost_per_unit:
                    lever_costs += lever_value * lever.cost_per_unit
        
        return (base_cost + lever_costs) * cost_multiplier
    
    def _calculate_roi(self, metrics: Dict[str, float], total_cost: float) -> float:
        """Calculate ROI for a trial."""
        if total_cost <= 0:
            return 0.0
        
        # Simple ROI calculation based on revenue
        revenue = metrics.get('revenue', 0.0)
        return (revenue - total_cost) / total_cost
    
    def _check_constraints(self, strategy: Strategy, trial_values: Dict[str, float], total_cost: float) -> List[str]:
        """Check if any constraints are violated."""
        violations = []
        
        for constraint in self.spec.constraints:
            if constraint.constraint_type == "budget":
                if constraint.operator == "le" and total_cost > constraint.value:
                    violations.append(f"Budget constraint violated: {total_cost:.2f} > {constraint.value}")
                elif constraint.operator == "ge" and total_cost < constraint.value:
                    violations.append(f"Budget constraint violated: {total_cost:.2f} < {constraint.value}")
                    
            elif constraint.constraint_type == "timeline":
                timeline = strategy.expected_timeline_months
                if constraint.operator == "le" and timeline > constraint.value:
                    violations.append(f"Timeline constraint violated: {timeline} > {constraint.value}")
                elif constraint.operator == "ge" and timeline < constraint.value:
                    violations.append(f"Timeline constraint violated: {timeline} < {constraint.value}")
        
        return violations
    
    def _aggregate_results(self, strategy: Strategy, trial_results: List[Dict], trials: int) -> SimulationResult:
        """Aggregate trial results into final simulation result."""
        # Extract all values for aggregation
        all_metrics = {metric.name: [] for metric in self.spec.metrics}
        all_costs = []
        all_rois = []
        all_violations = []
        
        for trial in trial_results:
            for metric_name, metric_value in trial['metrics'].items():
                all_metrics[metric_name].append(metric_value)
            all_costs.append(trial['total_cost'])
            all_rois.append(trial['roi'])
            all_violations.extend(trial['constraint_violations'])
        
        # Calculate statistics for each metric
        metrics_stats = {}
        for metric_name, values in all_metrics.items():
            if values:
                metrics_stats[metric_name] = self._calculate_statistics(values)
        
        # Calculate cost statistics
        cost_stats = self._calculate_statistics(all_costs)
        
        # Calculate ROI statistics
        roi_stats = self._calculate_statistics(all_rois)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(all_rois, all_violations)
        
        # Calculate success rate (no constraint violations)
        success_rate = 1.0 - (len(set(all_violations)) / trials)
        
        return SimulationResult(
            strategy_name=strategy.name,
            trials=trials,
            metrics=metrics_stats,
            total_cost=cost_stats,
            roi=roi_stats,
            risk_score=risk_score,
            constraint_violations=list(set(all_violations)),
            success_rate=success_rate
        )
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures for a list of values."""
        if not values:
            return {}
        
        values_array = np.array(values)
        
        # Calculate percentiles
        p10 = np.percentile(values_array, 10)
        p90 = np.percentile(values_array, 90)
        
        # Calculate CVaR (Conditional Value at Risk)
        cvar_threshold = 0.05  # 5% tail
        tail_values = values_array[values_array <= np.percentile(values_array, cvar_threshold * 100)]
        cvar = np.mean(tail_values) if len(tail_values) > 0 else np.min(values_array)
        
        return {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'p10': float(p10),
            'p90': float(p90),
            'cvar': float(cvar)
        }
    
    def _calculate_risk_score(self, rois: List[float], violations: List[str]) -> float:
        """Calculate a risk score based on ROI volatility and constraint violations."""
        if not rois:
            return 0.0
        
        # Base risk from ROI volatility
        roi_array = np.array(rois)
        roi_volatility = np.std(roi_array) / (np.mean(roi_array) + 1e-8)
        
        # Risk from constraint violations
        violation_penalty = len(set(violations)) * 0.1
        
        # Risk from negative ROI probability
        negative_roi_prob = np.mean(roi_array < 0)
        
        # Combine risk factors (0-10 scale)
        risk_score = min(10.0, roi_volatility * 5 + violation_penalty + negative_roi_prob * 5)
        
        return float(risk_score)
