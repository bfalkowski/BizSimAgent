# BizSimAgent

A standalone CLI tool that gamifies business asks by translating them into strategies, running simulations under uncertainty, and outputting a ranked leaderboard of options with costs, risks, ROI, and tradeoffs.

## Features

- **Business Ask Translation**: Converts plain text business requests into structured specifications
- **Strategy Generation**: Creates multiple candidate strategies based on decision levers
- **Monte Carlo Simulation**: Runs simulations under uncertainty with configurable priors
- **Gamified Leaderboard**: Ranks strategies with badges, scores, and risk metrics
- **Agent Mode**: Can be used as a WaaP agent for automated business analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd JiraGamification2

# Install in development mode
pip install -e .

# Or install with pip
pip install .
```

## Usage

### CLI Mode

BizSimAgent can be used as a standalone CLI tool with four main commands:

#### 1. Translate Business Ask
```bash
bizsim translate --ask "Launch usage-based billing for our SaaS platform" --out spec.json
```

#### 2. Generate Strategies
```bash
bizsim generate --spec spec.json --out strategies.json
```

#### 3. Run Simulations
```bash
bizsim simulate --spec spec.json --strategies strategies.json --out leaderboard.json --trials 1000
```

#### 4. Get Recommendations
```bash
bizsim recommend --leaderboard leaderboard.json --out rec.json
```

### Agent Mode

When used as a WaaP agent, BizSimAgent reads from `context.json` and produces structured outputs:

```bash
python -m agents.bizsim
```

Required inputs in `context.json`:
- `ask.text`: The business ask
- `constraints.json`: Business constraints
- `priors.json`: Prior distributions for simulation

Outputs:
- `/results/sim.leaderboard.json`: Simulation results and rankings
- `/results/sim.recommendation.json`: Recommended strategy with rationale
- `/artifacts/sim/`: Charts and visualizations

## Project Structure

```
JiraGamification2/
├── bizsim/
│   ├── __init__.py
│   ├── cli.py          # Typer CLI with 4 commands
│   ├── models.py       # Pydantic data models
│   ├── simulate.py     # Monte Carlo simulation engine
│   └── gamify.py       # Badge and scoring logic
├── agents/
│   └── bizsim.py       # WaaP agent wrapper
├── results/            # JSON outputs
├── artifacts/          # Charts and visualizations
├── pyproject.toml      # Project configuration
└── README.md           # This file
```

## Simulation Features

- **Distribution Types**: Normal, lognormal, beta, and triangular distributions
- **Risk Metrics**: Mean, P10/P90 percentiles, CVaR risk measures
- **Gamification**: Badges for performance, risk tolerance, and innovation
- **Uncertainty Handling**: Monte Carlo simulation with configurable trial counts

## Example Output

The tool generates structured JSON outputs including:
- Strategy rankings with scores
- Risk metrics and confidence intervals
- Cost-benefit analysis
- Trade-off assessments
- Performance badges and achievements

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black .
isort .

# Run tests
pytest
```

## License

MIT License
# start-a2a-helloworld
# start-a2a-helloworld
