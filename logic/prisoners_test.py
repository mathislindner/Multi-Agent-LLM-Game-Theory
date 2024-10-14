#from models.test_model import TestModel
from models.huggingface_models import get_huggingface_pipeline
from data.prompts.prisoners_dilemma_prompts import *
import os
import json

class baseAgent():
    def __init__(self, name, prompt):
        self.name = name
        self.prompt = [prompt]

    def append_prompt(self, message):
        self.prompt.append(message)

    def remove_prompt(self):
        return self.prompt.pop()

def parse_for_decision(generted_dict):
    content = generted_dict["content"]
    #look for COOPERATE or DEFECT
    if "COOPERATE" in content:
        return "COOPERATE"
    elif "DEFECT" in content:
        return "DEFECT"

def generate(prompt, pipe):
    return pipe(
        prompt,
        max_new_tokens=5,
    )[0]["generated_text"][-1]

def decision_message(decision, role):
    return {
        'role': role,
        'content': f'\n{role}: {decision}'
    }

def add_prompts_from_messages(agent1, agent2, agent1_message, agent2_message):
    agent1.append_prompt(decision_message(agent1_message, "assistant"))
    agent1.append_prompt(decision_message(agent2_message, "user"))
    agent2.append_prompt(decision_message(agent2_message, "assistant"))
    agent2.append_prompt(decision_message(agent1_message, "user"))

    agent1.append_prompt(call_for_decision)
    agent2.append_prompt(call_for_decision)

    return agent1, agent2

def evaluate_outcome(agent1_decision, agent2_decision):
    payoff_matrix = {
        ("COOPERATE", "COOPERATE"): (2, 2),
        ("COOPERATE", "DEFECT"): (0, 3),
        ("DEFECT", "COOPERATE"): (3, 0),
        ("DEFECT", "DEFECT"): (1, 1),
    }
    return payoff_matrix[(agent1_decision, agent2_decision)]
    
def launch_game(MODEL_ID):
    game_state = {}
    
    agent1 = baseAgent("agent 1", game_prompt)
    agent2 = baseAgent("agent 2", game_prompt)

    agent1.append_prompt(assistant_1_prompt)
    agent2.append_prompt(assistant_2_prompt)

    agent1.append_prompt(call_for_message)
    agent2.append_prompt(call_for_message)

    pipe = get_huggingface_pipeline(MODEL_ID)
    print("Game started")
    #generate messages
    agent1_message = generate(agent1.prompt, pipe)
    agent2_message = generate(agent2.prompt, pipe)

    #add messages from agents to prompts
    agent1, agent2 = add_prompts_from_messages(agent1_message, agent2_message)
    #let the agents decide
    agent1_decision = generate(agent1.prompt, pipe)
    agent2_decision = generate(agent2.prompt, pipe)

    def evaluate_outcome(agent1_decision, agent2_decision):
        payoff_matrix = {
            ("COOPERATE", "COOPERATE"): (2, 2),
            ("COOPERATE", "DEFECT"): (0, 3),
            ("DEFECT", "COOPERATE"): (3, 0),
            ("DEFECT", "DEFECT"): (1, 1),
        }
        return payoff_matrix[(agent1_decision, agent2_decision)]

    #parse decisions for COOPERATE or DEFECT
    agent1_decision = parse_for_decision(agent1_decision)
    agent2_decision = parse_for_decision(agent2_decision)

    print(json.dumps(agent1.prompt, indent=4))
    print(json.dumps(agent2.prompt, indent=4))
    print(agent1_message["content"])
    print(agent2_message["content"])
    print(agent1_decision["content"])
    print(agent2_decision["content"])
    
    outcome = evaluate_outcome(agent1_decision, agent2_decision)
    print(f"Outcome: {outcome}")

