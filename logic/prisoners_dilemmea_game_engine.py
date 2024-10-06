from agents.custom_agent import BaseAgent
from typing import Dict, Any

class GraphState:
    current_round: int = 0
    games: Dict[str, Any] = {} # round, agent1_intent, agent2_intent, agent1_decision, agent2_decision, payoffs
    agent1: BaseAgent = None
    agent2: BaseAgent = None

    def __init__(self, agent1: BaseAgent, agent2: BaseAgent):
        self.agent1 = agent1
        self.agent2 = agent2


def run_next_game(state: GraphState)-> Dict[str, Any]:



graph_state = GraphState()

first_game_result = run_game(graph_state)