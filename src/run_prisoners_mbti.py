
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command, Send
from typing import TypedDict, List, Annotated, Literal, Union
from pydantic import BaseModel
import operator
from src.models import get_model
import json
import pandas as pd
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from src.prompting.prisoners_dilemma_prompts import get_personality_from_key_prompt, get_game_description_prompt, get_game_history_as_messages_prompt, get_call_for_action, get_call_for_message
import re
# https://blog.langchain.dev/langgraph/
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/react-agent-structured-output.ipynb
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/map-reduce.ipynb

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
    current_round: int
    
    personality_key_1: str
    personality_key_2: str
    agent_1_messages: Annotated[List[str], operator.add]
    agent_1_actions: Annotated[List[str], operator.add]
    agent_1_scores: Annotated[List[int], operator.add]
    agent_2_messages: Annotated[List[str], operator.add]
    agent_2_actions: Annotated[List[str], operator.add]
    agent_2_scores: Annotated[List[int], operator.add]

def get_agent_annotated_prompt(agent_name: str, state: PDState, prompt_type: Literal["message", "action"]) -> AnnotatedPrompt:
    '''Get the prompt for the agent based on the state of the game. The prompt includes the agent's personality, the game history, and a call to action or message.
    Args:
        agent_name (str): The name of the agent
        state (PDState): The state of the game
        prompt_type (Literal["message", "action"]): The type of prompt to generate
    Returns:
        AnnotatedPrompt: The prompt for the agent
    '''
    prompt = []
    if agent_name == "agent_1":
        agent_prompt = get_personality_from_key_prompt(state["personality_key_1"])
    else:
        agent_prompt = get_personality_from_key_prompt(state["personality_key_2"])
    prompt.append(agent_prompt)
    history = get_game_history_as_messages_prompt(agent_name, state, prompt_type)
    prompt.extend(history)
    prompt.append(state["game_description_prompt"])
    if prompt_type == "message":
        prompt.append(get_call_for_message())
    else:
        prompt.append(get_call_for_action())
    return AnnotatedPrompt(agent_name=agent_name, prompt_type=prompt_type, prompt=prompt)

def send_prompts_node(prompt_type = Literal["message", "action"]):
    def send_prompts(state: PDState):
        agent_1_annotated_prompt_state = get_agent_annotated_prompt("agent_1", state, prompt_type)
        agent_2_annotated_prompt_state = get_agent_annotated_prompt("agent_2", state, prompt_type)
        return [Send(f"invoke_from_prompt_state_{prompt_type}", agent_1_annotated_prompt_state), Send(f"invoke_from_prompt_state_{prompt_type}", agent_2_annotated_prompt_state)]
    return send_prompts

def invoke_from_prompt_state_node(model, ActionResponse):
    def invoke_from_prompt_state(state : AnnotatedPrompt):
        prompt = state.prompt
        agent_name = state.agent_name
        prompt_type = state.prompt_type
        Structure = MessageResponse if prompt_type == "message" else ActionResponse
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
    return (state["current_round"] <= state["total_rounds"])

def run_n_rounds_w_com(model_name: str, total_rounds: int, personality_key_1: str, personality_key_2: str, game_name: str) -> PDState:
    # get models
    model = get_model(model_name)
    callback_handler = OpenAICallbackHandler()
    if game_name == "prisoners_dilemma":
        ActionResponse = PDActionResponse
    elif game_name == "stag_hunt":
        ActionResponse = SHActionResponse
    #create graph
    graph = StateGraph(PDState, input = PDState, output = PDState)
    #add nodes
    graph.add_node("lambda_to_messages", lambda x: {})
    graph.add_node("lambda_from_messages", lambda x: {})
    graph.add_node(f"invoke_from_prompt_state_message", invoke_from_prompt_state_node(model, ActionResponse))
    graph.add_node(f"invoke_from_prompt_state_action", invoke_from_prompt_state_node(model, ActionResponse))
    graph.add_node(f"lambda_to_actions", lambda x: {})
    graph.add_node("lambda_from_actions", lambda x: {})
    graph.add_node("update_state", update_state_node(game_name))
    
    #add edges
    graph.add_edge(START, "lambda_to_messages")
    graph.add_conditional_edges(
        source = "lambda_to_messages", 
        path = send_prompts_node("message"),
        path_map = ["invoke_from_prompt_state_message"]
        )
    graph.add_edge("invoke_from_prompt_state_message","lambda_from_messages")
    graph.add_conditional_edges(
        source = "lambda_from_messages", 
        path = send_prompts_node("action"),
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
    #print(compiled_graph.get_graph().draw_mermaid())
    #create initial state
    game_description_prompt = get_game_description_prompt(game_name) #get from data.prompts #say how many total rounds will be played
    initial_state = PDState(
        game_description_prompt=game_description_prompt,
        personality_key_1=personality_key_1,
        personality_key_2=personality_key_2,
        current_round=1,
        total_rounds=total_rounds
    )
    end_state = compiled_graph.invoke(initial_state, config={"recursion_limit": 200, "callbacks": [callback_handler]})
    print(f"Total Cost (USD): ${callback_handler.total_cost}")
    #save results in pd df
    path_to_csv = "/cluster/home/mlindner/Github/master_thesis_project/src/data/outputs/experiment_250309.csv"
    columns = ["model_name", "personality_1", "personality_2", "agent_1_scores", "agent_2_scores", "agent_1_messages", "agent_2_messages", "agent_1_actions", "agent_2_actions", "total_rounds", "total_tokens", "total_cost_USD"]

    #fix the messages and actions by removing quotes or double quotes if there are any at the start or end with regex
    def clean_text(text):
        return re.sub(r'^[\'"]|[\'"]$', '', text)

    end_state["agent_1_messages"] = [clean_text(msg) for msg in end_state["agent_1_messages"]]
    end_state["agent_2_messages"] = [clean_text(msg) for msg in end_state["agent_2_messages"]]
    end_state["agent_1_actions"] = [clean_text(action) for action in end_state["agent_1_actions"]]
    end_state["agent_2_actions"] = [clean_text(action) for action in end_state["agent_2_actions"]]
    # Create a new row with the results
    new_row = pd.DataFrame([{
        "model_name": model_name,
        "personality_1": personality_key_1,
        "personality_2": personality_key_2,
        "agent_1_scores": end_state["agent_1_scores"],
        "agent_2_scores": end_state["agent_2_scores"],
        "agent_1_messages": end_state["agent_1_messages"],
        "agent_2_messages": end_state["agent_2_messages"],
        "agent_1_actions": end_state["agent_1_actions"],
        "agent_2_actions": end_state["agent_2_actions"],
        "total_rounds": total_rounds,
        "total_tokens": callback_handler.total_tokens,
        "total_cost_USD": callback_handler.total_cost
    }])

    try:
        df = pd.read_csv(path_to_csv)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
    if df.empty:
        df = new_row
    else:
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path_to_csv, mode='w', header=True, index=False)
    return end_state