from flask import render_template, request, redirect, url_for, flash, jsonify
from app import app
import pokersim
from pokersim.game.spingo import SpinGoGame
from pokersim.agents.random_agent import RandomAgent
from pokersim.agents.rule_based_agent import RuleBased1Agent
from pokersim.agents.advanced_opponent_agent import AdvancedOpponentAgent, AdvancedOpponentProfile

@app.route('/')
def index():
    """Homepage with Poker Simulator Overview"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page with information about the simulator"""
    return render_template('about.html')

@app.route('/examples')
def examples():
    """Example simulations page"""
    return render_template('examples.html')

@app.route('/api/spin-and-go', methods=['POST'])
def api_spin_and_go():
    """API endpoint to simulate a Spin and Go tournament"""
    data = request.json
    buy_in = data.get('buy_in', 10)
    num_players = data.get('num_players', 3)
    
    # Create a Spin and Go tournament
    tournament = SpinGoGame(buy_in=buy_in, num_players=num_players)
    
    # Create agents (default to rule-based agents)
    agents = []
    for i in range(num_players):
        agent_type = data.get(f'agent_{i}_type', 'rule_based')
        if agent_type == 'random':
            agents.append(RandomAgent(i))
        elif agent_type == 'advanced':
            # Create an advanced agent with default profile
            agents.append(AdvancedOpponentAgent(i))
        else:
            # Default to rule-based
            agents.append(RuleBased1Agent(i))
    
    # Run a simplified simulation
    results = {
        'buy_in': buy_in,
        'num_players': num_players,
        'multiplier': tournament._determine_multiplier(),  # Access the multiplier
        'prize_pool': tournament.prize_pool,
        'player_agents': [agent.__class__.__name__ for agent in agents]
    }
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)