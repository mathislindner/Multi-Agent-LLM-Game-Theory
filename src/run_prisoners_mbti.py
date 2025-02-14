
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, List, Annotated, Literal
from pydantic import BaseModel
import operator
from src.models import get_model
from langchain_core.tools import tool
# https://blog.langchain.dev/langgraph/
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/react-agent-structured-output.ipynb

#inherites from MessagesState but we added better structured output for better readability
#messages: List[Dict[str, str]] = [] https://dev.to/jamesli/advanced-langgraph-implementing-conditional-edges-and-tool-calling-agents-3pdn
class PDState(MessagesState):
    """State for prisonner's dilemma game, includes actions taken and messages exchanged by agents.

    Args:
        TypedDict ([type]): [description]
    """
    game_description_prompt: str
    agent_1_personality: str
    agent_2_personality: str
    agent_1_messages: Annotated[List[str], operator.add]
    agent_1_actions: Annotated[List[str], operator.add]
    agent_2_messages: Annotated[List[str], operator.add]
    agent_2_actions: Annotated[List[str], operator.add]
    current_round: int
    total_rounds: int
    
    def __init__(self, game_description_prompt: str, agent_1_personality: str, agent_2_personality: str, total_rounds: int):
        self.game_description_prompt = game_description_prompt
        self.agent_1_personality = agent_1_personality
        self.agent_2_personality = agent_2_personality
        self.agent_1_messages = []
        self.agent_1_actions = []
        self.agent_2_messages = []
        self.agent_2_actions = []
        self.round = 1
        self.total_rounds = total_rounds
        
class ActionResponse(BaseModel):
    """Repond with action to take: cooperate or defect."""
    action: Literal["cooperate", "defect"]
    
class MessageResponse(BaseModel):
    """Respond with message to send to the other agent."""
    message: str

@tool
def take_action(action: Literal["cooperate", "defect"]) -> str:
    """Use this to choose if you are cooperating or defecting."""
    #TODO: make sure we know who is taking the action
    return action

@tool
def send_message(message: str) -> str:
    """Use this to send a message to the other agent."""
    #TODO: make sure we know who is sending the message
    return message

# Define the function that calls the model
def call_model_action_node(model, agent_name:str):
    def call_model_action(state: PDState) -> PDState:
        #TODO: use state to form a prompt
        response = model.with_structured_output(ActionResponse).invoke(state["messages"])
        #TODO: add to state
        return state
    return call_model_action

def call_model_message_node(model, agent_name:str):
    def call_model_message(state: PDState) -> PDState:
        #TODO: use state to form a prompt, using agent name and personality and what happened in the games.
        message_prompt = "" #
        response = model.with_structured_output(MessageResponse).invoke(message_prompt)
        #TODO: addto state
        return state
    return call_model_message

def increment_round(state: PDState) -> PDState:
    state["current_round"] += 1
    return state

def should_continue(current_round: int ,total_rounds : int) -> bool:
    return (total_rounds < current_round)

def run_n_rounds_w_com(model_name: str, rounds: int, personality_key_1: str, personality_key_2: str) -> None:
    # get models
    model = get_model(model_name)
    # create agents from strings
    
    # create graph from agents and n rounds
    #init graph
    game_description_prompt = ""
    personality_prompt_1 = ""
    personality_prompt_2 = ""
    
    graph = StateGraph(PDState) #add state definition
    graph.add_node("init_PD", PDState(game_description_prompt, personality_prompt_1, personality_prompt_2))
    graph.add_node(f"message_agent_1", call_model_message_node(model, "agent_1"))
    graph.add_node(f"message_agent_2", call_model_message_node(model, "agent_2"))
    graph.add_node(f"action_agent_1", call_model_action_node(model, "agent_1"))
    graph.add_node(f"action_agent_2", call_model_action_node(model, "agent_2"))
    graph.add_node(f"increment", increment_round)
    
    graph.add_edge(START, "init_PD")
    graph.add_edge("init_PD", "message_agent_1")
    graph.add_edge("init_PD", "message_agent_2")
    graph.add_edge("message_agent_1", "action_agent_1")
    graph.add_edge("message_agent_1", "action_agent_2")
    graph.add_edge("message_agent_2", "action_agent_1")
    graph.add_edge("message_agent_2", "action_agent_2")
    graph.add_edge("action_agent_1", "increment")
    graph.add_edge("action_agent_2", "increment")
    graph.add_conditional_edges(
        source = "message_agent_1",
        path = should_continue,
        path_map = {
            False : END,
            True : ["message_agent_1", "message_agent_2"]
            }
        )
    
    compiled_graph = graph.compile()
    #print mermaid
    print(compiled_graph.get_graph().draw_mermaid())
    
    return None


"""
    for i in range(1, rounds+1):
        # add nodes
        graph.add_node(f"round_{i}_message_agent_1", call_model_message_node(model, personality_names[0]))
        graph.add_node(f"round_{i}_message_agent_2", call_model_message_node(model, personality_names[1]))
        graph.add_node(f"round_{i}_action_agent_1", call_model_action_node(model, personality_names[0]))
        graph.add_node(f"round_{i}_action_agent_2", call_model_action_node(model, personality_names[1]))
        graph.add_node(f"round_{i}_increment", increment_round)
        # add edges
        graph.add_edge("start", f"round_{i}_message_agent_1")
        graph.add_edge("start", f"round_{i}_message_agent_2")
        
        graph.add_edge(f"round_{i}_message_agent_1", f"round_{i}_action_agent_1")
        graph.add_edge(f"round_{i}_message_agent_1", f"round_{i}_action_agent_2")
        graph.add_edge(f"round_{i}_message_agent_2", f"round_{i}_action_agent_1")
        graph.add_edge(f"round_{i}_message_agent_2", f"round_{i}_action_agent_2")
        
        graph.add_edge(f"round_{i}_action_agent_1", f"round_{i}_increment")
        graph.add_edge(f"round_{i}_action_agent_2", f"round_{i}_increment")      
    """