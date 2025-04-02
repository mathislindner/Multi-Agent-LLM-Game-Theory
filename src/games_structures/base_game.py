from abc import ABC, abstractmethod
from typing import Type, List, Union, Annotated, TypedDict, Literal
import operator
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class MessageResponse(BaseModel):
    """Respond with a sentence to send to the other agent."""
    message: str

class BaseGameStructure(ABC):
    """Abstract base class that enforces a complete game structure."""
    @property
    @abstractmethod
    def game_name(self) -> str:
        """Each game must have a name."""
        pass
    
    @property
    def MessageResponse(self) -> Type[BaseModel]:
        """Default message response, can be overridden by subclasses."""
        return MessageResponse
    
    @property
    @abstractmethod
    def ActionResponse(self) -> Type[BaseModel]:
        """Each game must define its own ActionResponse."""
        pass
    
    @property
    @abstractmethod
    def GAME_PROMPT(self) -> SystemMessage:
        """Each game must have a game prompt."""
        pass
    
    @property
    @abstractmethod
    def payoff_matrix(self) -> dict:
        """Each game must have a payoff matrix."""
        pass
    
    @property
    def coerce_message(self) -> SystemMessage:
        """force the agent to write a message"""
        return SystemMessage("According to the description, the game history, your personality, your instrinsic goals, write the message you want to send to the other agent now.")
    
    @property
    def coerce_action(self) -> SystemMessage:
        """force the agent to give an action"""
        return SystemMessage("According to the description, the game history, your personality, your last message and the other's agents message, give your action now")

class AnnotatedPrompt(BaseModel):
    agent_name: str
    prompt_type: Literal["message", "action"]
    prompt: List[Union[HumanMessage, SystemMessage, AIMessage]]
    
class GameState(TypedDict):
    """State for the games, includes actions taken and messages exchanged by agents.
    Args:
        TypedDict ([type]): [description]
    """
    total_rounds: int
    current_round: int
    
    personality_key_1: str 
    model_name_1: str
    agent_1_messages: Annotated[List[str], operator.add]
    agent_1_actions: Annotated[List[str], operator.add]
    intent_agent_1: Annotated[List[str], operator.add]
    truthful_agent_1: Annotated[List[str], operator.add]
    analysis_agent_1: Annotated[List[str], operator.add]
    agent_1_scores: Annotated[List[int], operator.add]
    
    personality_key_2: str
    model_name_2: str
    agent_2_messages: Annotated[List[str], operator.add]
    agent_2_actions: Annotated[List[str], operator.add]
    intent_agent_2: Annotated[List[str], operator.add]
    truthful_agent_2: Annotated[List[str], operator.add]
    analysis_agent_2: Annotated[List[str], operator.add]
    agent_2_scores: Annotated[List[int], operator.add]
    

def get_game_history(current_agent, state, history_type: str):
    '''return the history as a list of human and assistant messages (current agent is assistant, the other is human)'''
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
            # sum all scores until current round
            if round_num <= len(current_agent_scores) and round_num <= len(other_agent_scores):
                current_agent_total_score = sum(current_agent_scores[:round_num])
                other_agent_total_score = sum(other_agent_scores[:round_num])
                game_history.append(SystemMessage(f"Your total score {current_agent_total_score} : {other_agent_total_score} Their total score"))
    return game_history