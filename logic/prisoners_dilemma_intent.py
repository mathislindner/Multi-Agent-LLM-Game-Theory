#langgraph imports
from tools.tool_node import ToolNode

from models.test_model import TestModel
from agents.custom_agent import randomAgent, honestAgent, dishonestAgent
from tools.prisoners_dilemma import PrisonersDilemmaDecision, PrisonersDilemmaPayoff


agent1 = honestAgent()
agent2 = dishonestAgent()
agent_tools = [PrisonersDilemmaDecision]

tool_node = ToolNode(tools)

model = 
def run_game(self):
    # Simulate thePrisonersDilemmaPayoff game, coordinating between agents
    agent1_message = self.agent1.communicate()
    agent2_message = self.agent2.communicate()
    
    agent1_decision = self.agent1.decide(agent2_message)
    agent2_decision = self.agent2.decide(agent1_message)
    
    outcome = (agent1_decision, agent2_decision)
    self.evaluate_outcome(outcome)

def evaluate_outcome(self, outcome):
    payoffs = self.payoff_matrix[outcome]
    print(f"Outcome: {outcome}, Payoffs: {payoffs}")
