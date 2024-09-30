#langgraph imports
from tools.tool_node import ToolNode

from models.test_model import TestModel
from agents.custom_agent import NormalAgent
from tools.prisoners_dilemma_tools import PrisonersDilemmaDecision, PrisonersDilemmaPayoff

agent1 = NormalAgent(name="agent 1", personality="You are an honest agent that always tells the truth.")
agent2 = NormalAgent(name="agent 2", personality="You are a dishonest agent that always lies.")
agent_tools = [PrisonersDilemmaDecision]

tool_node = ToolNode(agent_tools)

model = TestModel(agent1, agent2, tool_node)

history = {}
"""
{
    "Game nr": 1,
    "agent1_intent": "cooperate",
    "agent2_intent": "cooperate",
    "agent1_decision": "cooperate",
    "agent2_decision": "defect"
    "payoffs": {
        "agent1": 0,
        "agent2": 3
    },
    Game nr: 2,
    ...
}
"""
def run_game(self):
    # Simulate thePrisonersDilemmaPayoff game, coordinating between agents
    agent1_intent = self.agent1.intent(history)
    agent2_intent = self.agent2.intent(history)



    outcome = (agent1_decision, agent2_decision)
    self.evaluate_outcome(outcome)

def evaluate_outcome(self, outcome):
    payoffs = self.payoff_matrix[outcome]
    print(f"Outcome: {outcome}, Payoffs: {payoffs}")
