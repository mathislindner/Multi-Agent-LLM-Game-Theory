
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command
from typing import TypedDict, List, Annotated, Literal, Union
from pydantic import BaseModel
import operator
from src.models import get_model
from langchain_core.tools import tool

from src.data.prompts.prisoners_dilemma_prompts import get_personality_from_key_prompt, get_game_description_prompt, get_game_history_as_messages_prompt, get_call_for_action, get_call_for_message
# https://blog.langchain.dev/langgraph/
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/react-agent-structured-output.ipynb

#inherites from MessagesState but we added better structured output for better readability
#messages: List[Dict[str, str]] = [] https://dev.to/jamesli/advanced-langgraph-implementing-conditional-edges-and-tool-calling-agents-3pdn

class PromptState(BaseModel):
    agent_name: str
    prompt: List[Union[HumanMessage, SystemMessage, AIMessage]]

class PromptsState(BaseModel):
    prompt_states: List[PromptState]

class LLMReponseState(BaseModel):
    agent_name: str
    response: str
    
class LLMReponseState(BaseModel):
    LLM_response_states: List[LLMReponseState]
        
class ActionResponse(BaseModel):
    """Repond with action to take: cooperate or defect."""
    action: Literal["cooperate", "defect"]
    
class MessageResponse(BaseModel):
    """Respond with message to send to the other agent."""
    message: str
    
class PDState(TypedDict):
    """State for prisonner's dilemma game, includes actions taken and messages exchanged by agents.

    Args:
        TypedDict ([type]): [description]
    """
    game_description_prompt: str
    total_rounds: int
    personality_key_1: str
    personality_key_2: str
    agent_1_messages: Annotated[List[str], operator.add]
    agent_1_actions: Annotated[List[str], operator.add]
    agent_1_scores: Annotated[List[int], operator.add]
    agent_2_messages: Annotated[List[str], operator.add]
    agent_2_actions: Annotated[List[str], operator.add]
    agent_2_scores: Annotated[List[int], operator.add]
    
    current_round: int
    current_message_prompts: Annotated[List[PromptsState], operator.add] #TODO need to clear these at the end of each round
    current_action_prompts: Annotated[List[PromptsState], operator.add]
    

def get_agent_message_promptstate(agent_name: str, state: PDState) -> PromptState:
    message_prompt = []
    if agent_name == "agent_1":
        agent_prompt = get_personality_from_key_prompt(state["personality_key_1"])
    else:
        agent_prompt = get_personality_from_key_prompt(state["personality_key_2"])
    message_prompt.append(agent_prompt)
    message_history = get_game_history_as_messages_prompt(agent_name, state, "message")
    message_prompt.extend(message_history)
    message_prompt.append(get_call_for_message())
    return PromptState(agent_name=agent_name, prompt=message_prompt, prompt_type="message")

def get_agent_action_promptstate(agent_name: str, state: PDState) -> PromptState:
    action_prompt = []
    if agent_name == "agent_1":
        agent_prompt = get_personality_from_key_prompt(state["personality_key_1"])
    else:
        agent_prompt = get_personality_from_key_prompt(state["personality_key_2"])
    action_prompt.append(agent_prompt)
    action_history = get_game_history_as_messages_prompt(agent_name, state, "action")
    action_prompt.extend(action_history)
    action_prompt.append(get_call_for_action())
    return PromptState(agent_name=agent_name, prompt=action_prompt, prompt_type="action")

def generate_messages_prompts(state: PDState):
    agent_1_message_state = get_agent_message_promptstate("agent_1", state)
    agent_2_message_state = get_agent_message_promptstate("agent_2", state)
    return {"current_message_prompts" : MessagesState(message_prompt_states=[agent_1_message_state, agent_2_message_state])}

def generate_actions_prompts(state: PDState):
    agent_1_action_state = get_agent_action_promptstate("agent_1", state)
    agent_2_action_state = get_agent_action_promptstate("agent_2", state)
    return {"current_action_prompts" : MessagesState(message_prompt_states=[agent_1_action_state, agent_2_action_state])}



#TODO: think abou thow we acn identify who sent what, since we can't just add it to a list witohut knowing sender 
def invoke_from_prompt_state(state: PromptState):
    prompt = state.prompt
    agent_name = state.agent_name
    response = model.with_structured_output(MessageResponse).invoke(prompt)
    message = response.message
    return {f"{agent_name}_messages": [message]}



def update_scores_node():
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
    
def increment_round_node():
    def increment_round(state: PDState) -> PDState:
        state["current_round"] += 1
        return state
    return increment_round

def should_continue(state: PDState) -> bool:
    return (state["current_round"] <= state["total_rounds"])


def run_n_rounds_w_com(model_name: str, total_rounds: int, personality_key_1: str, personality_key_2: str) -> None:
    # get models
    model = get_model(model_name)
    #create graph
    graph = StateGraph(PDState, input = PDState, output = PDState)
    #add nodes
    graph.add_node(f"map", lambda x: x) #This just exists, because you cannot map from a conditional edge
    graph.add_node(f"message_agent_1", call_model_message_node(model, "agent_1"))
    graph.add_node(f"message_agent_2", call_model_message_node(model, "agent_2"))
    graph.add_node(f"reduce_and_map", reduce_and_map_node) #to
    graph.add_node(f"action_agent_1", call_model_action_node(model, "agent_1"))
    graph.add_node(f"action_agent_2", call_model_action_node(model, "agent_2"))
    graph.add_node(f"update_scores", update_scores_node())
    graph.add_node(f"increment", increment_round_node())
    
    #add edges
    graph.add_edge(START, "map")
    graph.add_edge("map", "message_agent_1")
    graph.add_edge("map", "message_agent_2")
    graph.add_edge("message_agent_1", "action_agent_1")
    graph.add_edge("message_agent_1", "action_agent_2")
    graph.add_edge("message_agent_2", "action_agent_1")
    graph.add_edge("message_agent_2", "action_agent_2")
    graph.add_edge("action_agent_1", "update_scores")
    graph.add_edge("action_agent_2", "update_scores")
    graph.add_edge("update_scores", "increment")
    graph.add_conditional_edges(
        source = "increment",
        path = should_continue,
        path_map = {
            False : END,
            True : "map"
            }
        )
    #compile and run
    compiled_graph = graph.compile()
    #print mermaid
    print(compiled_graph.get_graph().draw_mermaid())
    #create initial state
    game_description_prompt = get_game_description_prompt() #get from data.prompts #say how many total rounds will be played
    initial_state = PDState(
        game_description_prompt=game_description_prompt,
        personality_key_1=personality_key_1,
        personality_key_2=personality_key_2,
        agent_1_messages=[],
        agent_1_actions=[],
        agent_1_scores=[],
        agent_2_messages=[],
        agent_2_actions=[],
        agent_2_scores=[],
        current_round=1,
        total_rounds=total_rounds
    )
    end_state = compiled_graph.invoke(initial_state)
    return end_state