
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, List, Annotated, Literal
from pydantic import BaseModel
import operator
from src.models import get_model
from langchain_core.tools import tool

from src.data.prompts.prisoners_dilemma_prompts import get_personality_from_key_prompt, get_game_description_prompt, get_game_history_prompt, get_call_for_action, get_call_for_message
# https://blog.langchain.dev/langgraph/
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/react-agent-structured-output.ipynb

#inherites from MessagesState but we added better structured output for better readability
#messages: List[Dict[str, str]] = [] https://dev.to/jamesli/advanced-langgraph-implementing-conditional-edges-and-tool-calling-agents-3pdn
class PDState(TypedDict):
    """State for prisonner's dilemma game, includes actions taken and messages exchanged by agents.

    Args:
        TypedDict ([type]): [description]
    """
    game_description_prompt: str
    personality_key_1: str
    personality_key_2: str
    agent_1_messages: Annotated[List[str], operator.add]
    agent_1_actions: Annotated[List[str], operator.add]
    agent_1_scores: Annotated[List[int], operator.add]
    agent_2_messages: Annotated[List[str], operator.add]
    agent_2_actions: Annotated[List[str], operator.add]
    agent_2_scores: Annotated[List[int], operator.add]
    current_round: int
    total_rounds: int
        
class ActionResponse(BaseModel):
    """Repond with action to take: cooperate or defect."""
    action: Literal["cooperate", "defect"]
    
class MessageResponse(BaseModel):
    """Respond with message to send to the other agent."""
    message: str

# Define the function that calls the model
def call_model_action_node(model, agent_name:str):
    def call_model_action(state: PDState) -> PDState:
        action_prompt: str = ""
        action_prompt += state["game_description_prompt"]
        action_prompt += get_game_history_prompt(agent_name, state["agent_1_messages"], state["agent_1_actions"], state["agent_2_messages"], state["agent_2_actions"], state["agent_1_scores"], state["agent_2_scores"], state["current_round"])
        if agent_name == "agent_1":
            action_prompt += get_personality_from_key_prompt(state["personality_key_1"])
        else:
            action_prompt += get_personality_from_key_prompt(state["personality_key_2"])
        action_prompt += get_call_for_action()
        response = model.with_structured_output(ActionResponse).invoke(action_prompt)
        #update state
        state[f"{agent_name}_actions"].append(response["action"])
        return state
    return call_model_action

def call_model_message_node(model, agent_name:str):
    def call_model_message(state: PDState) -> PDState:
        message_prompt = ""
        message_prompt += state["game_description_prompt"]
        if agent_name == "agent_1":
            message_prompt += get_personality_from_key_prompt(state["personality_key_1"])
        else:
            message_prompt += get_personality_from_key_prompt(state["personality_key_2"])
        message_prompt += get_game_history_prompt(state["agent_1_messages"], state["agent_1_actions"], state["agent_2_messages"], state["agent_2_actions"], state["current_round"])
        message_prompt += get_call_for_message()
        response = model.with_structured_output(MessageResponse).invoke(message_prompt)
        if agent_name == "agent_1":
            state["agent_1_messages"].append(response["message"])
        else:
            state["agent_2_messages"].append(response["message"])
        return state
    return call_model_message

def update_scores_node() -> None:
    def update_scores(state: PDState) -> PDState:
        payoff_matrix = {
            ("cooperate", "cooperate"): (3, 3),
            ("cooperate", "defect"): (0, 5),
            ("defect", "cooperate"): (5, 0),
            ("defect", "defect"): (1, 1),
        }
        agent_1_decision = state["agent_1_actions"][-1]
        agent_2_decision = state["agent_2_actions"][-1]
        score_agent1, score_agent2 = payoff_matrix[(agent_1_decision, agent_2_decision)]
        state["agent_1_scores"].append(score_agent1)
        state["agent_2_scores"].append(score_agent2)
        return state
    return update_scores
    

def increment_round(state: PDState) -> PDState:
    state["current_round"] += 1
    return state

def should_continue(state: PDState) -> bool:
    return (state["current_round"] <= state["total_rounds"])

def run_n_rounds_w_com(model_name: str, total_rounds: int, personality_key_1: str, personality_key_2: str) -> None:
    # get models
    model = get_model(model_name)
    #create graph
    graph = StateGraph(PDState)
    #add nodes
    graph.add_node(f"distribute", lambda x: x) #This just exists, because you cannot distribute from a conditional edge
    graph.add_node(f"message_agent_1", call_model_message_node(model, "agent_1"))
    graph.add_node(f"message_agent_2", call_model_message_node(model, "agent_2"))
    graph.add_node(f"action_agent_1", call_model_action_node(model, "agent_1"))
    graph.add_node(f"action_agent_2", call_model_action_node(model, "agent_2"))
    graph.add_node(f"update_scores", update_scores_node)
    graph.add_node(f"increment", increment_round)
    
    #add edges
    graph.add_edge(START, "distribute")
    graph.add_edge("distribute", "message_agent_1")
    graph.add_edge("distribute", "message_agent_2")
    graph.add_edge("message_agent_1", "action_agent_1")
    graph.add_edge("message_agent_1", "action_agent_2")
    graph.add_edge("message_agent_2", "action_agent_1")
    graph.add_edge("message_agent_2", "action_agent_2")
    graph.add_edge("action_agent_1", "gather")
    graph.add_edge("action_agent_2", "gather")
    graph.add_edge("gather", "increment")
    graph.add_conditional_edges(
        source = "increment",
        path = should_continue,
        path_map = {
            False : END,
            True : "distribute"
            }
        )
    #compile and run
    compiled_graph = graph.compile()
    #print mermaid
    print(compiled_graph.get_graph().draw_mermaid())
    #create initial state
    game_description_prompt = "" #get from data.prompts #say how many total rounds will be played
    initial_state = {"game_description_prompt":game_description_prompt, "personality_key_1":personality_key_1, "personality_key_2":personality_key_2, "current_round" : 1, "total_rounds" : total_rounds}
    end_state = compiled_graph.invoke(initial_state)
    return end_state