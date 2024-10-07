from models.test_model import TestModel
from data.prompts.prisoners_dilemma_prompts import *

class baseAgent():
    name : str = "None"
    prompt : str = ""

    def __init__(self, name, prompt):
        self.name = name
        self.append_prompt(prompt)

    def append_prompt(self, message):
        self.prompt += message
        
def launch_game():
    game_state = {}

    llm = TestModel()

    agent1 = baseAgent("agent1", game_prompt)
    agent2 = baseAgent("agent2", game_prompt)

    agent1.append_prompt(agent_1_prompt)
    agent2.append_prompt(agent_2_prompt)

    agent1.append_prompt(altruist_agent_prompt)
    agent2.append_prompt(selfish_agent_prompt)

    agent1.append_prompt(call_for_message + "\n agent 1 (you): ")
    agent2.append_prompt(call_for_message + "\n agent 2 (you): ")

    agent1_message = llm.invoke(agent1.prompt)
    agent2_message = llm.invoke(agent2.prompt)

    def add_prompts_from_messages(agent1_message, agent2_message):
        reply_agent_1 = agent1_message.split("you): ")[-1]
        reply_agent_2 = agent2_message.split("you): ")[-1]
        agent1.append_prompt(reply_agent_2 + call_for_decision +"\n agent 1 (you): ")
        agent2.append_prompt(reply_agent_1 + call_for_decision +"\n agent 2 (you): ")

    add_prompts_from_messages(agent1_message, agent2_message)

    agent1_decision = llm.invoke(agent1.prompt).split("you): ")[-1]
    agent2_decision = llm.invoke(agent2.prompt).split("you): ")[-1]

    def evaluate_outcome(agent1_decision, agent2_decision):
        payoff_matrix = {
            ("cooperate", "cooperate"): (2, 2),
            ("cooperate", "defect"): (0, 3),
            ("defect", "cooperate"): (3, 0),
            ("defect", "defect"): (1, 1),
        }
        return payoff_matrix[(agent1_decision, agent2_decision)]

    outcome = evaluate_outcome(agent1_decision, agent2_decision)

    print(f"Agent 1 decided to {agent1_decision}")
    print(f"Agent 2 decided to {agent2_decision}")
    print(f"Outcome: {outcome}")

