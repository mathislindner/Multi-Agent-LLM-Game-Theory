from abc import ABC, abstractmethod
from typing import Type
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class MessageResponse(BaseModel):
    """Respond with a sentence to send to the other agent."""
    message: str

class BaseGameStructure(ABC):
    """Abstract base class that enforces a complete game structure."""

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
