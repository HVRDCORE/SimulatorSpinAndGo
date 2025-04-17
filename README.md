# PokerSim: Python Poker Simulation Framework

A comprehensive Python framework for poker simulation, machine learning research, and game theory exploration, featuring advanced opponent modeling and performance optimizations.

## Features

- **Game Engine**: Complete Texas Hold'em poker engine with various betting structures
- **Agent System**: Multiple agent types from basic to advanced ML-based agents
- **ML Integration**: PyTorch integration for deep learning and neural networks
- **Performance Optimization**: High-performance computing with GPU acceleration
- **Flexible API**: Easy-to-use interface for custom agent development

## Quick Start

1. **Installation**
   All dependencies are automatically installed through pyproject.toml.

2. **Running Examples**
   - Basic game simulation:
   ```bash
   python examples/basic_game.py
   ```
   
   - Advanced usage examples:
   ```bash
   python examples/advanced_usage.py --all
   ```
   
   - Train an ML agent:
   ```bash
   python examples/train_ppo.py
   ```

3. **Web Interface**
   Start the web interface for visualization and monitoring:
   ```bash
   python main.py
   ```
   Access the interface at: https://{repl-name}.{username}.repl.co

## Project Structure

- `pokersim/`: Core framework modules
- `examples/`: Example scripts and tutorials
- `docs/`: Documentation and API reference
- `tests/`: Unit and integration tests

## Usage Examples

```python
from pokersim.game.state import GameState
from pokersim.agents.random_agent import RandomAgent

# Create a simple game with random agents
game = GameState(num_players=2)
agents = [RandomAgent() for _ in range(2)]

# Run a single hand
game.play_hand(agents)
```

## Documentation

- Full API documentation: See `docs/api_reference.md`
- Getting started guide: See `docs/getting_started.md`
- Example tutorials: Browse the `examples/` directory

## Contributing

Feel free to fork this template and customize it for your needs. Bug reports and pull requests are welcome.
