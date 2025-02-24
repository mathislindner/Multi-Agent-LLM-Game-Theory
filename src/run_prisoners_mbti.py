
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command, Send
from typing import TypedDict, List, Annotated, Literal, Union
from pydantic import BaseModel
import operator
from src.models import get_model
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
import json

from src.prompting.prisoners_dilemma_prompts import get_personality_from_key_prompt, get_game_description_prompt, get_game_history_as_messages_prompt, get_call_for_action, get_call_for_message
# https://blog.langchain.dev/langgraph/
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/react-agent-structured-output.ipynb
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/map-reduce.ipynb
#inherites from MessagesState but we added better structured output for better readability
#messages: List[Dict[str, str]] = [] https://dev.to/jamesli/advanced-langgraph-implementing-conditional-edges-and-tool-calling-agents-3pdn

class AnnotatedPrompt(BaseModel):
    agent_name: str
    prompt_type: Literal["message", "action"]
    prompt: List[Union[HumanMessage, SystemMessage, AIMessage]]

class LLMReply(BaseModel):
    agent_name: str
    reply_type: Literal["message", "action"]
    message: str
    

#These are for the models structured output
class PDActionResponse(BaseModel):
    """Repond with action to take: cooperate or defect."""
    action: Literal["cooperate", "defect"]
    
class SHActionResponse(BaseModel):
    """Repond with action to take: stag or hare."""
    action: Literal["stag", "hare"]
    
class MessageResponse(BaseModel):
    """Respond with a sentence to send to the other agent."""
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
    
    #current_message_responses: Annotated[list, operator.add]
    #current_action_responses: Annotated[list, operator.add]
    
def get_agent_message_annotated_prompt(agent_name: str, state: PDState) -> AnnotatedPrompt:
    message_prompt = []
    message_prompt.append(state["game_description_prompt"])
    if agent_name == "agent_1":
        agent_prompt = get_personality_from_key_prompt(state["personality_key_1"]) #TODO could rename key to easily access it without if
    else:
        agent_prompt = get_personality_from_key_prompt(state["personality_key_2"])
    message_prompt.append(agent_prompt)
    message_history = get_game_history_as_messages_prompt(agent_name, state, "message")
    message_prompt.extend(message_history)
    message_prompt.append(get_call_for_message())
    return AnnotatedPrompt(agent_name=agent_name, prompt_type="message", prompt=message_prompt)

def get_agent_action_annotated_prompt(agent_name: str, state: PDState) -> AnnotatedPrompt:
    action_prompt = []
    action_prompt.append(state["game_description_prompt"])
    if agent_name == "agent_1":
        agent_prompt = get_personality_from_key_prompt(state["personality_key_1"])
    else:
        agent_prompt = get_personality_from_key_prompt(state["personality_key_2"])
    action_prompt.append(agent_prompt)
    action_history = get_game_history_as_messages_prompt(agent_name, state, "action")
    action_prompt.extend(action_history)
    action_prompt.append(get_call_for_action())
    return AnnotatedPrompt(agent_name=agent_name, prompt_type="action", prompt=action_prompt)

def send_messages_prompts(state: PDState):
    agent_1_annotated_prompt_state = get_agent_message_annotated_prompt("agent_1", state)
    agent_2_annotated_prompt_state = get_agent_message_annotated_prompt("agent_2", state)
    # this should send directly to invoke from state prompt
    return [Send("invoke_from_prompt_state_message", agent_1_annotated_prompt_state), Send("invoke_from_prompt_state_message", agent_2_annotated_prompt_state)]
    #return {"current_message_prompts" : [agent_1_annotated_prompt_state, agent_2_annotated_prompt_state]}

def send_actions_prompts(state: PDState):
    agent_1_annotated_prompt_state = get_agent_action_annotated_prompt("agent_1", state)
    agent_2_annotated_prompt_state = get_agent_action_annotated_prompt("agent_2", state)
    return [Send("invoke_from_prompt_state_action", agent_1_annotated_prompt_state), Send("invoke_from_prompt_state_action", agent_2_annotated_prompt_state)]

def invoke_from_prompt_state_node(model, ActionResponse):
    def invoke_from_prompt_state(state : AnnotatedPrompt):
        prompt = state.prompt
        agent_name = state.agent_name
        prompt_type = state.prompt_type
        Structure = MessageResponse if prompt_type == "message" else ActionResponse
        print("invoking model...")
        response = model.with_structured_output(Structure).invoke(prompt)
        message = response.message if prompt_type == "message" else response.action #TODO this is ugly but it helps for the model to understand it s working with an action
        return Command(update = {f"{agent_name}_{prompt_type}s": [message]})
    return invoke_from_prompt_state

def update_state_node(game_name: str):
    def update_state(state: PDState) -> PDState:
        state_updates = {}
        payoff_matrix = {}
        with (open("/cluster/home/mlindner/Github/master_thesis_project/src/prompting/payoff_matrices.json")) as f:
            payoff_matrix = json.load(f)[game_name]
        
        # update scores
        agent_1_decision = state["agent_1_actions"][-1]
        agent_2_decision = state["agent_2_actions"][-1]
        score_agent1, score_agent2 = payoff_matrix[agent_1_decision][agent_2_decision]
        
        # add scores to scores
        state_updates["agent_1_scores"] = [score_agent1]
        state_updates["agent_2_scores"] = [score_agent2]
        
        #increment round
        state_updates["current_round"] = state["current_round"] + 1 
        return Command(update = state_updates)
    return update_state

def should_continue(state: PDState) -> bool:
    print("should continue??")
    return (state["current_round"] <= state["total_rounds"])
    
def test1(state: PDState) -> dict:
    return {}
    
def test2(state: PDState) -> dict:
    return {}

def test3(state: PDState) -> dict:
    return {}

def test4(state: PDState) -> dict:
    return {}

def run_n_rounds_w_com(model_name: str, total_rounds: int, personality_key_1: str, personality_key_2: str, game_name: str) -> PDState:
    # get models
    model = get_model(model_name)
    if game_name == "prisoners_dilemma":
        ActionResponse = PDActionResponse
    elif game_name == "stag_hunt":
        ActionResponse = SHActionResponse
    #create graph
    graph = StateGraph(PDState, input = PDState, output = PDState)
    #add nodes
    graph.add_node("lambda_to_messages", test1)
    graph.add_node("lambda_from_messages", test2)
    graph.add_node(f"invoke_from_prompt_state_message", invoke_from_prompt_state_node(model, ActionResponse)) #doing this because else we fall into a weird recursion state
    graph.add_node(f"invoke_from_prompt_state_action", invoke_from_prompt_state_node(model, ActionResponse))
    graph.add_node(f"lambda_to_actions", test3)
    graph.add_node("lambda_from_actions", test4)
    graph.add_node("update_state", update_state_node(game_name))
    
    #add edges
    graph.add_edge(START, "lambda_to_messages")
    graph.add_conditional_edges(
        source = "lambda_to_messages", 
        path = send_messages_prompts,
        path_map = ["invoke_from_prompt_state_message"]
        )
    graph.add_edge("invoke_from_prompt_state_message","lambda_from_messages")
    #graph.add_edge("gather_messages", "generate_actions_prompts")
    graph.add_conditional_edges(
        source = "lambda_from_messages", 
        path = send_actions_prompts,
        path_map = ["invoke_from_prompt_state_action"]
        )
    graph.add_edge("invoke_from_prompt_state_action","lambda_from_actions")
    graph.add_edge("lambda_from_actions", "update_state")
    graph.add_conditional_edges(
        source = "update_state",
        path = should_continue,
        path_map = {
            False : END,
            True : "lambda_to_messages"
            }
        )
    #compile and run
    compiled_graph = graph.compile()
    #print mermaid
    print(compiled_graph.get_graph().draw_mermaid())
    #create initial state
    game_description_prompt = get_game_description_prompt(game_name) #get from data.prompts #say how many total rounds will be played
    initial_state = PDState(
        game_description_prompt=game_description_prompt,
        personality_key_1=personality_key_1,
        personality_key_2=personality_key_2,
        current_round=1,
        total_rounds=total_rounds
    )
    end_state = compiled_graph.invoke(initial_state, config={"recursion_limit": 200})
    return end_state