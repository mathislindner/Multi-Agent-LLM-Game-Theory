#from models.test_model import TestModel
from models.huggingface_models import get_huggingface_pipeline
from data.prompts.prisoners_dilemma_prompts import *
import os

class baseAgent():
    name : str = "None"
    prompt : list = []

    def __init__(self, name, prompt):
        self.name = name
        self.append_prompt(prompt)

    def append_prompt(self, message):
        self.prompt.append(message)

    def remove_prompt(self):
        self.prompt.pop()
        
def launch_game(MODEL_ID):
    game_state = {}

    pipe = get_huggingface_pipeline(MODEL_ID)
    #processor = AutoProcessor.from_pretrained(model_id)

    agent1 = baseAgent("agent1", game_prompt)
    agent2 = baseAgent("agent2", game_prompt)

    agent1.append_prompt(agent_1_prompt)
    agent2.append_prompt(agent_2_prompt)

    agent1.append_prompt(call_for_message)
    agent2.append_prompt(call_for_message)

    print("Game started")

    agent1_message = pipe(
        agent1.prompt,
        max_new_tokens=20,
    )[0]["generated_text"][-1]
    
    print(f"Agent 1 sent the message: {agent1_message}")

    agent2_message = pipe(
        text_inputs = agent2.prompt,
        max_new_tokens = 20,
    )[0]["generated_text"][-1]
    print(f"Agent 2 sent the message: {agent2_message}")

    #TODO add to game state
    game_state["agent1_message"] = agent1_message
    game_state["agent2_message"] = agent2_message

    def add_prompts_from_messages(agent1_message, agent2_message):
        #TODO
        reply_agent_1 = {"role":"agent 1", "content": agent1_message}
        reply_agent_2 = {"role":"agent 2", "content": agent2_message}
        agent_1_history = {agent1.remove_prompt() + reply_agent_1}
        agent_2_history = {agent2.remove_prompt() + reply_agent_2}
        agent1.append_prompt(reply_agent_2)
        agent1.append_prompt(reply_agent_1)
        agent2.append_prompt(reply_agent_1)
        agent2.append_prompt(reply_agent_2)
        agent1.append_prompt(call_for_decision)
        agent2.append_prompt(call_for_decision)

    add_prompts_from_messages(agent1_message, agent2_message)

    #agent1_decision = llm.invoke(agent1.prompt).split("you): ")[-1]
    #agent2_decision = llm.invoke(agent2.prompt).split("you): ")[-1]

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

