from pydantic import BaseModel
from typing import Literal
from src.games_structures.base_game import BaseGameStructure
from langchain_core.messages import SystemMessage

# Game: Deadlock
game_prompt_deadlock = '''**You are a player in a repeated Deadlock game with communication.**
## Instructions:
Each round, you choose to **Cooperate** or **Defect**. Communication is allowed before deciding.
## Outcomes:
- **Both defect**: Both get 2 points.
- **One defects, one cooperates**: Defector gets 3, Cooperator gets 0.
- **Both cooperate**: Both get 1.
## Notes:
- Your goal is to get the highest score possible.'''

class DeadlockActionResponse(BaseModel):
    """Respond with action to take: cooperate or defect."""
    action: Literal["cooperate", "defect"]

class DeadlockGame(BaseGameStructure):
    """Structured class for the Deadlock game."""

    @property
    def game_name(self):
        return "deadlock"

    @property
    def ActionResponse(self):
        return DeadlockActionResponse

    @property
    def GAME_PROMPT(self):
        return SystemMessage(game_prompt_deadlock)

    @property
    def payoff_matrix(self):
        return {
            ("cooperate", "cooperate"): (1, 1),
            ("cooperate", "defect"): (0, 3),
            ("defect", "cooperate"): (3, 0),
            ("defect", "defect"): (2, 2)
        }
