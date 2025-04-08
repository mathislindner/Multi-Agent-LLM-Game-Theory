from abc import ABC, abstractmethod
from typing import Type, List, Annotated, TypedDict
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
        #TODO: not the best spot to ave this: probably should be a helper.
        """force the agent to write a message"""
        return SystemMessage("According to the description, the game history, your personality, your instrinsic goals, write the message you want to send to the other agent now. json") # added the word json to use a json mode and it requires to mention the word in the prompt 
    
    @property
    def coerce_action(self) -> SystemMessage:
        """force the agent to give an action"""
        return SystemMessage("According to the description, the game history, your personality, your last message and the other's agents message, give your action now. json")  # added the word json to use a json mode and it requires to mention the word in the prompt 
    
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