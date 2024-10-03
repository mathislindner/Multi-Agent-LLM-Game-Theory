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
communication_tool_node = ToolNode(prisoners_dilemma_communication)
decision_tool_node = ToolNode(prisoners_dilemma_decision)

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
workflow.add_node("agent1_comm", communication_tool_node)
workflow.add_node("agent2_comm", communication_tool_node)
workflow.add_node("agent1_decision", decision_tool_node)
workflow.add_node("agent2_decision", decision_tool_node)

# Add the increment_round function as a node
workflow.add_node("increment_round", increment_round)

# Set the entry point for agent1 communication in the first round
workflow.add_edge(START, "agent1_comm")

# Set the entry point for agent2 communication in the first round
workflow.add_edge(START, "agent2_comm")

# Agent1 communication, Agent2 communication -> agent1_decision #TODO somehow pass the communcations to the prompt s.t. the agent can add the decision to the state
workflow.add_edge("agent1_comm", "agent1_decision")
workflow.add_edge("agent2_comm", "agent1_decision")

# Agent1 communication, Agent2 communication -> agent2_decision #TODO somehow pass the communcations to the prompt s.t. the agent can add the decision to the state
workflow.add_edge("agent1_comm", "agent2_decision")
workflow.add_edge("agent2_comm", "agent2_decision")

# Agent2 decision, Agent1 decision -> increment_round
workflow.add_edge("agent1_decision", "increment_round")
workflow.add_edge("agent2_decision", "increment_round")

# Compile the workflow
app = workflow.compile()

#draw the langgraph
print(app.get_graph().draw_mermaid())


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
