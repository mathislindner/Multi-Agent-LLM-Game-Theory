
from langgraph.graph import StateGraph
from typing import TypedDict, List, Annotated
import operator
from models import get_model

# https://blog.langchain.dev/langgraph/
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/react-agent-structured-output.ipynb

class PD_State(TypedDict):
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
    all_actions: Annotated[List[str], operator.add]


def run_n_rounds_w_com(model_name, rounds, mbti_strings) -> None:
    # get models
    model = get_model(model_name)
    # create agents from strings
    
    # create graph from agents and n rounds
    graph = StateGraph(PD_State) #add state definition
    graph.add_node(0, PD_State(input="start", agent_1_messages=[], agent_1_actions=[], agent_2_messages=[], agent_2_actions=[], all_actions=[]))
    for i in range(1, rounds+1):
        pass
    return None
        
    
    
    # launch graph with n nodes
    