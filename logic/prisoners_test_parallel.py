from models.huggingface_models import get_huggingface_pipeline
from data.prompts.prisoners_dilemma_prompts import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def generate_message(pipe, agent, max_tokens=20):
    return pipe(
        agent.prompt,
        max_new_tokens=max_tokens,
    )[0]["generated_text"][-1]

def launch_game(MODEL_ID):
    game_state = {}

    pipe = get_huggingface_pipeline(MODEL_ID)

    agent1 = baseAgent("agent1", game_prompt)
    agent2 = baseAgent("agent2", game_prompt)

    agent1.append_prompt(agent_1_prompt)
    agent2.append_prompt(agent_2_prompt)

    agent1.append_prompt(call_for_message)
    agent2.append_prompt(call_for_message)

    print("Game started")

    # Use ThreadPoolExecutor to run both agents in parallel for generating messages
    with ThreadPoolExecutor() as executor:
        future_agent1_message = executor.submit(generate_message, pipe, agent1)
        future_agent2_message = executor.submit(generate_message, pipe, agent2)

        agent1_message = future_agent1_message.result()
        agent2_message = future_agent2_message.result()

    print(f"Agent 1 sent the message: {agent1_message}")
    print(f"Agent 2 sent the message: {agent2_message}")

    # Add messages to game state
    game_state["agent1_message"] = agent1_message
    game_state["agent2_message"] = agent2_message

    def add_prompts_from_messages(agent1_message, agent2_message):
        reply_agent_1 = {"role": "agent 1", "content": agent1_message}
        reply_agent_2 = {"role": "agent 2", "content": agent2_message}
        agent_1_history = {agent1.remove_prompt() + reply_agent_1}
        agent_2_history = {agent2.remove_prompt() + reply_agent_2}
        agent1.append_prompt(reply_agent_2)
        agent1.append_prompt(reply_agent_1)
        agent2.append_prompt(reply_agent_1)
        agent2.append_prompt(reply_agent_2)
        agent1.append_prompt(call_for_decision)
        agent2.append_prompt(call_for_decision)

    # Sync before decisions are made
    add_prompts_from_messages(agent1_message, agent2_message)

    # Parallelize decision making
    with ThreadPoolExecutor() as executor:
        future_agent1_decision = executor.submit(generate_message, pipe, agent1, max_tokens=5)
        future_agent2_decision = executor.submit(generate_message, pipe, agent2, max_tokens=5)

        agent1_decision = future_agent1_decision.result()
        agent2_decision = future_agent2_decision.result()

    print(f"Agent 1 decided to {agent1_decision}")
    print(f"Agent 2 decided to {agent2_decision}")

    def evaluate_outcome(agent1_decision, agent2_decision):
        payoff_matrix = {
            ("cooperate", "cooperate"): (2, 2),
            ("cooperate", "defect"): (0, 3),
            ("defect", "cooperate"): (3, 0),
            ("defect", "defect"): (1, 1),
        }
        return payoff_matrix.get((agent1_decision, agent2_decision), (0, 0))  # Default case if decision is unexpected

    outcome = evaluate_outcome(agent1_decision, agent2_decision)

    print(f"Outcome: {outcome}")

# Assuming the following function is being used to start the game:
# launch_game("MODEL_ID")
