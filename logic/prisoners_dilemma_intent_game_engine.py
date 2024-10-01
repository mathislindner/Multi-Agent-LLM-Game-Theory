import json
from datetime import datetime
from typing import Optional, Literal

from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from tools.prisoners_dilemma_tools import prisoners_dilemma_communication, prisoners_dilemma_decision
from models.test_model import TestModel
from agents.custom_agent import baseAgent

# Define the agents and their personalities
agent1 = baseAgent(name="agent 1", personality="You are an honest agent that always tells the truth.")
agent2 = baseAgent(name="agent 2", personality="You are a dishonest agent that always lies.")

# Tools for communication and decision
agent_tools = [prisoners_dilemma_communication, prisoners_dilemma_decision]
tool_node = ToolNode(agent_tools)

# Define a TestModel to simulate agent behavior
model = TestModel()

# Game State Configuration
MAX_ROUNDS = 10
current_round = 1

# Helper to evaluate the outcome
def evaluate_outcome(agent1_decision, agent2_decision):
    payoff_matrix = {
        ("cooperate", "cooperate"): (2, 2),
        ("cooperate", "defect"): (0, 3),
        ("defect", "cooperate"): (3, 0),
        ("defect", "defect"): (1, 1),
    }
    return payoff_matrix[(agent1_decision, agent2_decision)]

# Function for handling the round transition
def increment_round(state: MessagesState) -> Literal["agent1_comm", END]:
    """Increments the game round and determines whether the game should continue."""
    global current_round
    if current_round >= MAX_ROUNDS:
        return END
    current_round += 1
    return "agent1_comm"

# LangGraph nodes and transitions
def call_model(state: MessagesState):
    """Function to call the agent's model."""
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(MessagesState)

# Define the sequence of states for communication and decision-making
workflow.add_node("agent1_comm", tool_node)
workflow.add_node("agent1_decision", tool_node)
workflow.add_node("agent2_comm", tool_node)
workflow.add_node("agent2_decision", tool_node)

# Set the entry point for agent1 communication in the first round
workflow.add_edge(START, "agent1_comm")

# Agent1 communication -> decision
workflow.add_edge("agent1_comm", "agent1_decision")

# Agent1 decision -> Agent2 communication
workflow.add_edge("agent1_decision", "agent2_comm")

# Agent2 communication -> decision
workflow.add_edge("agent2_comm", "agent2_decision")

# Agent2 decision -> End of round or new round
workflow.add_conditional_edges("agent2_decision", increment_round)

# Compile the workflow
app = workflow.compile()

# Game Runner
def run_game():
    global current_round
    history = {}

    while current_round <= MAX_ROUNDS:
        print(f"Starting round {current_round}")

        #TODO: Agent 1 and Agent 2 communicate their intent to each other

        #TODO: Agent 1 and Agent 2 execute an action based on the intent and personality

        # Evaluate the outcome of the round
        payoffs = evaluate_outcome(agent1_decision, agent2_decision)
        round_data = {
            "agent1_intent": agent1_intent,
            "agent2_intent": agent2_intent,
            "agent1_decision": agent1_decision,
            "agent2_decision": agent2_decision,
            "payoffs": {
                "agent1": payoffs[0],
                "agent2": payoffs[1]
            }
        }
        history[f"Game {current_round['round']}"] = round_data

        # Print round outcome
        print(f"Round {current_round['round']} completed with payoffs: {payoffs}")

    # Save the history
    datetime_str = datetime.now().strftime("%y%m%d_%H%M%S")
    with open(f"game_histories/prisoners_dilemma/{datetime_str}.json", "w") as f:
        json.dump(history, f, indent=4)

    return history
