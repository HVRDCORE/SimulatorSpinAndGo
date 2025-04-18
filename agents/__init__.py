"""
Agents module containing different poker-playing agents.

This module includes base agent classes and various implementations of poker agents,
from simple rule-based agents to advanced opponents with sophisticated strategies.
"""

from pokersim.agents.base_agent import Agent
from pokersim.agents.random_agent import RandomAgent
from pokersim.agents.call_agent import CallAgent
from pokersim.agents.rule_based_agent import RuleBased1Agent, RuleBased2Agent
from pokersim.agents.advanced_opponent_agent import (
    AdvancedOpponentAgent, AdvancedOpponentProfile,
    LOOSE_AGGRESSIVE_PROFILE, TIGHT_PASSIVE_PROFILE, 
    LOOSE_PASSIVE_PROFILE, TIGHT_AGGRESSIVE_PROFILE,
    BALANCED_PROFILE, MANIAC_PROFILE, ROCK_PROFILE, ADAPTIVE_PROFILE
)

__all__ = [
    "Agent", "RandomAgent", "CallAgent", "RuleBased1Agent", "RuleBased2Agent",
    "AdvancedOpponentAgent", "AdvancedOpponentProfile",
    "LOOSE_AGGRESSIVE_PROFILE", "TIGHT_PASSIVE_PROFILE", 
    "LOOSE_PASSIVE_PROFILE", "TIGHT_AGGRESSIVE_PROFILE",
    "BALANCED_PROFILE", "MANIAC_PROFILE", "ROCK_PROFILE", "ADAPTIVE_PROFILE"
]
