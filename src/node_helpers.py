import json
from typing import get_args, Literal, List, Union, Type
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.games_structures.base_game import BaseGameStructure, GameState

class AnnotatedPrompt(BaseModel):
    agent_name: str
    prompt_type: Literal["message", "action"]
    prompt: List[Union[HumanMessage, SystemMessage, AIMessage]]

def load_game_structure_from_registry(game_name: str) -> BaseGameStructure:
    '''Load the game structure from the registry based on the game name.
    Args:
        game_name (str): The name of the game
    Returns:
        BaseGameStructure: The game structure object
    '''
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

def get_game_history(current_agent, state, history_type: str):
    '''return the history as a list of human and assistant messages (current agent is assistant, the other is human)
    Args:
        current_agent (str): The name of the current agent
        state (GameState): The state of the game
        history_type (str): The type of history to return, either "message" or "action"
    Returns:
        List[Union[HumanMessage, SystemMessage, AIMessage]]: The game history as a list of messages
    '''
    if history_type not in ["message", "action"]:
        raise ValueError("history_type can only be 'message' or 'action'")

    agent_1_messages = state['agent_1_messages']
    agent_1_actions = state['agent_1_actions']
    agent_2_messages = state['agent_2_messages']
    agent_2_actions = state['agent_2_actions']
    agent_1_scores = state['agent_1_scores']
    agent_2_scores = state['agent_2_scores']
    current_round = state['current_round']

    game_history = []
    if current_agent == "agent_1":
        agent_1_message_type = AIMessage
        agent_2_message_type = HumanMessage
        current_agent_scores = agent_1_scores
        other_agent_scores = agent_2_scores
    else:
        agent_1_message_type = HumanMessage
        agent_2_message_type = AIMessage
        current_agent_scores = agent_2_scores
        other_agent_scores = agent_1_scores

    for round_num in range(1, current_round + 1):
        if round_num <= len(agent_1_messages) and round_num <= len(agent_2_messages):
            game_history.append(agent_1_message_type(agent_1_messages[round_num - 1]))
            game_history.append(agent_2_message_type(agent_2_messages[round_num - 1]))
            if round_num <= len(agent_1_actions) and round_num <= len(agent_2_actions):
                game_history.append(agent_1_message_type(agent_1_actions[round_num - 1]))
                game_history.append(agent_2_message_type(agent_2_actions[round_num - 1]))
            if round_num <= len(current_agent_scores) and round_num <= len(other_agent_scores):
                current_agent_total_score = sum(current_agent_scores[:round_num])
                other_agent_total_score = sum(other_agent_scores[:round_num])
                game_history.append(HumanMessage(f"Your total score {current_agent_total_score} : {other_agent_total_score} Their total score")) #TODO changed this to tool bc of claude
    return game_history

def get_personality_from_key_prompt(personality_key:str) -> SystemMessage:
    personalities = json.load(open("src/prompting/mbti_prompts_250129.json"))
    return SystemMessage(personalities[personality_key])

def get_answer_format(game_structure: BaseGameStructure):
    '''return the answer format to judge the action and message of a specific game
    Args:
        game_structure (BaseGameStructure): The game structure object
    Returns:
        AnswerFormat: The answer format for the game
    '''
    action_response_cls = game_structure.ActionResponse
    action_type = action_response_cls.__annotations__["action"]

    class AnswerFormat(BaseModel):
        analysis: str
        answer: action_type

    return AnswerFormat

def get_question_prompt(game_structure):
    """Get the question prompt for the game.
    Args:
        game_structure (BaseGameStructure): The game structure object.
    Returns:
        str: The question prompt string.
    """
    action_response_cls = game_structure.ActionResponse
    action_type = action_response_cls.__annotations__["action"]
    action_1, action_2 = get_args(action_type)
    return (
        f"You have to assess the intent of the person using the message that will be given. "
        f"To do that, first, do a one sentence analysis of the message, "
        f"if it seems like the person is looking to {action_1} or {action_2}."
    )
    
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
        agent_prompt = get_personality_from_key_prompt(state["personality_key_1"]) #system
    else:
        agent_prompt = get_personality_from_key_prompt(state["personality_key_2"]) #system
    prompt.append(agent_prompt)
    history = get_game_history(agent_name, state, prompt_type) #not only system prompts
    
    if (len(history)>0):
        prompt.append(SystemMessage("The following are the previous interactions"))
        prompt.extend(history) #moved this for claude
    else:
        prompt.append(SystemMessage("No history for now, this is a new game."))
        
    prompt.append(GameStructure.GAME_PROMPT)
    if prompt_type == "message":
        prompt.append(GameStructure.coerce_message) #changed for Anthropic to humanmessage
    else:
        prompt.append(GameStructure.coerce_action)
    return AnnotatedPrompt(agent_name=agent_name, prompt_type=prompt_type, prompt=prompt)


from langchain_core.utils.json_schema import dereference_refs
def generate_dereferenced_schema(model: Type[BaseModel]) -> dict:
    """
    Returns a JSON schema for the given Pydantic model where references
    like `#/defs/Name` are fully dereferenced (inlined).
    This is required for Gemini VertexAI to work because their protobuf based schema
    doesn't support nested references. The returned json objects can be converted
    back into the original pydantic objects because they will have identical json
    structure.
    """
    raw_schema = model.model_json_schema(ref_template="#/defs/{model}")
    if "$defs" in raw_schema:
        raw_schema["defs"] = raw_schema.pop("$defs")

    inlined = dereference_refs(raw_schema)
    inlined.pop("defs", None)
    return inlined

def merge_following_system_prompts(prompts : List[Union[HumanMessage, SystemMessage, AIMessage]]):
    """
    Only merges prompts that are of type SystemMessage and are following each other
    """
    merged = []
    current_content = []
    for prompt in prompts:
        if isinstance(prompt, SystemMessage):
            current_content.append(prompt.content)
        else:
            if current_content:
                merged.append(SystemMessage(content="".join(current_content)))
                current_content = []
            merged.append(prompt)
    if current_content:
        merged.append(HumanMessage(content=".")) #if there is not other message than a system message, s.t. anthropic does not throw an error
        merged.append(SystemMessage(content="".join(current_content)))
    return merged
