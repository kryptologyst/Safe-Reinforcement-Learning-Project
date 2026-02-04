# Safe Reinforcement Learning Project

This project implements Safe Reinforcement Learning algorithms with a focus on constrained policy optimization and safety-aware exploration. 

## WARNING: NOT FOR PRODUCTION CONTROL

This project is designed for research and educational purposes only. It should NOT be used for controlling real-world systems, especially in safety-critical domains such as:
- Autonomous vehicles
- Medical devices
- Industrial control systems
- Financial trading systems
- Any system where failure could result in harm

## Features

- **Safe RL Algorithms**: Constrained Policy Optimization (CPO), Lagrangian methods, penalty-based reward shaping
- **Safety Metrics**: Constraint violations, CVaR analysis, cost tracking
- **Modern Stack**: Gymnasium, PyTorch 2.x, structured logging, type hints
- **Interactive Demo**: Streamlit-based visualization and policy evaluation
- **Comprehensive Evaluation**: Statistical significance testing, ablation studies

## Quick Start

1. Install dependencies:
```bash
pip install -e .
pip install -e ".[dev]"  # For development
```

2. Run training:
```bash
python scripts/train.py --config-name=safe_cpo
```

3. Launch interactive demo:
```bash
streamlit run demo/app.py
```

## Project Structure

```
src/
├── algorithms/          # Safe RL algorithm implementations
├── policies/            # Policy networks and architectures
├── environments/        # Custom environments and wrappers
├── evaluation/          # Evaluation metrics and utilities
├── utils/              # Common utilities and helpers
└── configs/            # Configuration files

configs/                # Hydra configuration files
scripts/                # Training and evaluation scripts
tests/                  # Unit tests
demo/                   # Streamlit demo application
assets/                 # Generated plots, videos, and artifacts
```

## Safety Considerations

- All algorithms include deterministic seeding for reproducibility
- Safety constraints are explicitly defined and monitored
- Evaluation includes comprehensive safety metrics
- Clear disclaimers about research-only usage

## Contributing

1. Install pre-commit hooks: `pre-commit install`
2. Run tests: `pytest`
3. Format code: `black src/ tests/`
4. Lint code: `ruff check src/ tests/`

## License

MIT License - See LICENSE file for details.
# Safe-Reinforcement-Learning-Project
