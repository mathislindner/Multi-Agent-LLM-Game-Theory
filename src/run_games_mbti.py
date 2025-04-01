from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send
from typing import Literal
from src.models import get_model_by_id_and_provider
import pandas as pd
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from src.prompting.personality_prompts import get_personality_from_key_prompt
from src.games_structures.base_game import BaseGameStructure, GameState, AnnotatedPrompt, get_game_history

# https://blog.langchain.dev/langgraph/
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/react-agent-structured-output.ipynb
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/map-reduce.ipynb

#TODO: move this function to a helper or sth...
def load_game_structure_from_registry(game_name: str) -> BaseGameStructure:
    if game_name == "prisoners_dilemma":
        from src.games_structures.prisonnersdilemma import PrisonersDilemmaGame
        return PrisonersDilemmaGame()
    elif game_name == "stag_hunt":
        from src.games_structures.staghunt import StagHuntGame
        return StagHuntGame()
    elif game_name == "generic":
        from src.games_structures.generic import GenericGame
        return GenericGame()
    elif game_name == "chicken":
        from src.games_structures.chicken import ChickenGame
        return ChickenGame()
    elif game_name == "coordination":
        from src.games_structures.coordination import CoordinationGame
        return CoordinationGame()
    elif game_name == "hawk_dove":
        from src.games_structures.hawk_dove import HawkDoveGame
        return HawkDoveGame()
    elif game_name == "deadlock":
        from src.games_structures.deadlock import DeadlockGame
        return DeadlockGame()
    elif game_name == "battle_of_sexes":
        from src.games_structures.battle_of_sexes import BattleOfSexesGame
        return BattleOfSexesGame()
    else:
        raise ValueError(f"Unknown game name: {game_name}")

def get_agent_annotated_prompt(agent_name: str, state: GameState, prompt_type: Literal["message", "action"], GameStructure: BaseGameStructure) -> AnnotatedPrompt:
    '''Get the prompt for the agent based on the state of the game. The prompt includes the agent's personality, the game history, and a call to action or message.
    Args:
        agent_name (str): The name of the agent
        state (GameState): The state of the game
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
    history = get_game_history(agent_name, state, prompt_type)
    prompt.extend(history)
    prompt.append(GameStructure.GAME_PROMPT)
    if prompt_type == "message":
        prompt.append(GameStructure.coerce_message)
    else:
        prompt.append(GameStructure.coerce_action)
    return AnnotatedPrompt(agent_name=agent_name, prompt_type=prompt_type, prompt=prompt)

def send_prompts_node(prompt_type : Literal["message", "action"], GameStructure: BaseGameStructure):
    def send_prompts(state: GameState):
        agent_1_annotated_prompt_state = get_agent_annotated_prompt("agent_1", state, prompt_type, GameStructure)
        agent_2_annotated_prompt_state = get_agent_annotated_prompt("agent_2", state, prompt_type, GameStructure)
        return [Send(f"invoke_from_prompt_state_{prompt_type}", agent_1_annotated_prompt_state), Send(f"invoke_from_prompt_state_{prompt_type}", agent_2_annotated_prompt_state)]
    return send_prompts

def invoke_from_prompt_state_node(models, GameStructure):
    def invoke_from_prompt_state(state : AnnotatedPrompt):
        prompt = state.prompt
        agent_name = state.agent_name
        prompt_type = state.prompt_type
        model = models[agent_name]
        Structure = GameStructure.MessageResponse if prompt_type == "message" else GameStructure.ActionResponse
        response = model.with_structured_output(Structure).invoke(prompt)
        message = response.message if prompt_type == "message" else response.action #TODO this is ugly but it helps for the model to understand it s working with an action
        return Command(update = {f"{agent_name}_{prompt_type}s": [message]})
    return invoke_from_prompt_state

def update_state_node(GameStructure):
    def update_state(state: GameState) -> Command:
        state_updates = {}
        # update scores
        agent_1_decision = state["agent_1_actions"][-1]
        agent_2_decision = state["agent_2_actions"][-1]
        score_agent1, score_agent2 = GameStructure.payoff_matrix[(agent_1_decision, agent_2_decision)]
        
        # add scores to scores
        state_updates["agent_1_scores"] = [score_agent1]
        state_updates["agent_2_scores"] = [score_agent2]
        
        #increment round
        state_updates["current_round"] = state["current_round"] + 1 
        return Command(update = state_updates)
    return update_state

def should_continue(state: GameState) -> bool:
    return (state["current_round"] <= state["total_rounds"])

def run_n_rounds_w_com(model_provider_1: str, model_name_1: str, model_provider_2: str, model_name_2: str, total_rounds: int, personality_key_1: str, personality_key_2: str, game_name: str, file_path: str) -> GameState:
    # get models
    models = {
        "agent_1": get_model_by_id_and_provider(model_name_1, provider=model_provider_1),
        "agent_2": get_model_by_id_and_provider(model_name_2, provider=model_provider_2)
    }
    callback_handler = OpenAICallbackHandler() #TODO verify that this does not throw errors if we don t use openai
    
    GameStructure = load_game_structure_from_registry(game_name) #game now includes the game prompt, the payoff matrix, the message response the action response formats
    
    #create graph
    graph = StateGraph(GameState, input = GameState, output = GameState)
    #add nodes
    graph.add_node("lambda_to_messages", lambda x: {})
    graph.add_node("lambda_from_messages", lambda x: {})
    graph.add_node(f"invoke_from_prompt_state_message", invoke_from_prompt_state_node(models, GameStructure))
    graph.add_node(f"invoke_from_prompt_state_action", invoke_from_prompt_state_node(models, GameStructure))
    graph.add_node(f"lambda_to_actions", lambda x: {})
    graph.add_node("lambda_from_actions", lambda x: {})
    graph.add_node("update_state", update_state_node(GameStructure))
    
    #add edges
    graph.add_edge(START, "lambda_to_messages")
    graph.add_conditional_edges(
        source = "lambda_to_messages", 
        path = send_prompts_node("message", GameStructure),
        path_map = ["invoke_from_prompt_state_message"]
        )
    graph.add_edge("invoke_from_prompt_state_message","lambda_from_messages")
    graph.add_conditional_edges(
        source = "lambda_from_messages", 
        path = send_prompts_node("action", GameStructure),
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
    initial_state = GameState(
        personality_key_1=personality_key_1,
        personality_key_2=personality_key_2,
        current_round=1,
        total_rounds=total_rounds
    )
    end_state = compiled_graph.invoke(initial_state, config={"recursion_limit": 200, "callbacks": [callback_handler]})
    print(f"Total Cost (USD): ${callback_handler.total_cost}")
    #save results in pd df
    path_to_csv = file_path
    columns = ["model_provider_1", "model_name_1", "model_provider_2", "model_name_2", "personality_1", "personality_2", "agent_1_scores", "agent_2_scores", "agent_1_messages", "agent_2_messages", "agent_1_actions", "agent_2_actions", "total_rounds", "total_tokens", "total_cost_USD"]

    end_state["agent_1_messages"] = [msg.replace('"', "'") for msg in end_state["agent_1_messages"]]
    end_state["agent_2_messages"] = [msg.replace('"', "'") for msg in end_state["agent_2_messages"]]
    end_state["agent_1_actions"] = [action.replace('"', "'") for action in end_state["agent_1_actions"]]
    end_state["agent_2_actions"] = [action.replace('"', "'") for action in end_state["agent_2_actions"]]
    #TODO: add game name
    new_row = pd.DataFrame([{
        "game_name": game_name,
        "model_provider_1": model_provider_1,
        "model_name_1": model_name_1,
        "model_provider_2": model_provider_2,
        "model_name_2": model_name_2,
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
        df = new_row
    else:
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path_to_csv, mode='w', header=True, index=False)
    return end_state