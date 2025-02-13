
from langgraph.graph import StateGraph, MessagesState
from typing import TypedDict, List, Annotated, Literal
import operator
from models import get_model
from langchain_core.tools import tool
# https://blog.langchain.dev/langgraph/
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/react-agent-structured-output.ipynb

#inherites from MessagesState but we added better structured output for better readability
class PD_State(MessagesState):
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
    round: int
    
    def __init__(self, game_description_prompt: str, agent_1_personality: str, agent_2_personality: str):
        self.game_description_prompt = game_description_prompt
        self.agent_1_personality = agent_1_personality
        self.agent_2_personality = agent_2_personality
        self.agent_1_messages = []
        self.agent_1_actions = []
        self.agent_2_messages = []
        self.agent_2_actions = []
        self.round = 1
        
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
    def call_model_action(state: PD_State):
        #TODO: use state to form a prompt
        # who called thi function? agent 1 or agent 2?
        response = model.with_structured_output(ActionResponse).invoke(state["messages"])
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
    return call_model_action

def call_model_message_node(model, agent_name:str):
    def call_model_message(state: PD_State):
        message_prompt = "" #
        response = model.with_structured_output(MessageResponse).invoke(message_prompt)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
    return call_model_message

def increment_round(state: PD_State):
    state["round"] += 1
    return state

def run_n_rounds_w_com(model_name: str, rounds: int, personality_names: List[str] ) -> None:
    # get models
    model = get_model(model_name)
    # create agents from strings
    
    # create graph from agents and n rounds
    #init graph
    game_description_prompt = ""
    personality_prompt_1 = ""
    personality_prompt_2 = ""
    graph = StateGraph(PD_State) #add state definition
    graph.add_node("start", PD_State(game_description_prompt, personality_prompt_1, personality_prompt_2))
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
        
    # run graph
    
    return None