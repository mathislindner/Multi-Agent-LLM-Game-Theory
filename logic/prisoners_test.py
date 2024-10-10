#from models.test_model import TestModel
from models.huggingface_models import get_huggingface_pipeline
from data.prompts.prisoners_dilemma_prompts import *
import os
import json

class baseAgent():
    name : str = "None"
    prompt : list = []

    def __init__(self, name, prompt):
        self.name = name
        self.append_prompt(prompt)

    def append_prompt(self, message):
        self.prompt.append(message)

    def remove_prompt(self):
        return self.prompt.pop()

def launch_game(MODEL_ID):
    game_state = {}
    
    agent1 = baseAgent("agent 1", game_prompt_1)
    agent2 = baseAgent("agent 2", game_prompt_2)

    agent1.append_prompt(agent_1_prompt)
    agent2.append_prompt(agent_2_prompt)

    agent1.append_prompt(call_for_message)
    agent2.append_prompt(call_for_message)

    print("Game started")
    pipe = get_huggingface_pipeline(MODEL_ID)
    agent1_message = pipe(
        agent1.prompt,
        max_new_tokens=20,
    )[0]["generated_text"][-1]

    agent1_message = json.loads(agent1_message['content'])    
    print(agent1_message)

    agent2_message = pipe(
        text_inputs = agent2.prompt,
        max_new_tokens = 20,
    )[0]["generated_text"][-1]

    agent2_message = json.loads(agent2_message['content'])
    print(agent2_message)

    #TODO add to game state
    game_state["agent1_message"] = agent1_message
    game_state["agent2_message"] = agent2_message

    def add_prompts_from_messages(agent1_message, agent2_message):
        #TODO
        _ = agent1.remove_prompt()
        _ = agent2.remove_prompt()

        agent1.append_prompt(agent1_message)
        agent1.append_prompt(agent2_message)
        agent2.append_prompt(agent2_message)
        agent2.append_prompt(agent1_message)

        agent1.append_prompt(call_for_decision)
        agent2.append_prompt(call_for_decision)

    add_prompts_from_messages(agent1_message, agent2_message)

    #print prompts
    print(agent1.prompt)
    print(agent2.prompt)

    agent1_decision = pipe(
        agent1.prompt,
        max_new_tokens=5,
    )[0]["generated_text"][-1]

    agent2_decision = pipe(
        agent2.prompt,
        max_new_tokens=5,
    )[0]["generated_text"][-1]

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

