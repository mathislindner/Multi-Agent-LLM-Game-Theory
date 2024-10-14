#from models.test_model import TestModel
from models.huggingface_models import get_huggingface_pipeline
from data.prompts.prisoners_dilemma_prompts import game_prompt, assistant_1_prompt, assistant_2_prompt, call_for_message, call_for_decision
from agents.custom_agent import baseAgent
from logic.pris_helpers import parse_for_decision, generate, decision_message, add_prompts_from_messages, evaluate_outcome
import os
import json

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
    agent1, agent2 = add_prompts_from_messages(agent1, agent2, agent1_message, agent2_message, call_for_decision)
    #let the agents decide
    agent1_decision = generate(agent1.prompt, pipe)
    agent2_decision = generate(agent2.prompt, pipe)

    #parse decisions for COOPERATE or DEFECT
    agent1_decision_outcome = parse_for_decision(agent1_decision)
    agent2_decision_outcome = parse_for_decision(agent2_decision)

    print(json.dumps(agent1.prompt, indent=4))
    print(json.dumps(agent2.prompt, indent=4))
    print(agent1_message["content"])
    print(agent2_message["content"])
    print(agent1_decision["content"])
    print(agent2_decision["content"])
    
    outcome = evaluate_outcome(agent1_decision_outcome, agent2_decision_outcome)
    print(f"Outcome: {outcome}")

