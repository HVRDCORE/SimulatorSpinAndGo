# PokerSim API Reference

This document provides a comprehensive reference for the PokerSim API, describing all the modules, classes, and functions available in the framework.

## Table of Contents

1. [Game Module](#game-module)
   - [Card](#card)
   - [Deck](#deck)
   - [GameState](#gamestate)
   - [HandEvaluator](#handevaluator)
2. [Agents Module](#agents-module)
   - [Base Agent](#base-agent)
   - [Simple Agents](#simple-agents)
   - [Rule-Based Agents](#rule-based-agents)
3. [Machine Learning Module](#machine-learning-module)
   - [Models](#models)
   - [Torch Integration](#torch-integration)
   - [TensorFlow Integration](#tensorflow-integration)
   - [Advanced Agents](#advanced-agents)
4. [Algorithms Module](#algorithms-module)
   - [Deep CFR](#deep-cfr)
   - [PPO](#ppo)
5. [Utilities Module](#utilities-module)
   - [Optimization](#optimization)
   - [Numba Optimizations](#numba-optimizations)
   - [Logging](#logging)

## Game Module

The game module provides the core functionality for simulating poker games.

### Card

```python
from pokersim.game.card import Card, Suit, Rank
