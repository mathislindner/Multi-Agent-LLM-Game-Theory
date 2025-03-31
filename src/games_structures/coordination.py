from pydantic import BaseModel
from typing import Literal
from src.games_structures.base_game import BaseGameStructure
from langchain_core.messages import SystemMessage

# Game: Coordination
game_prompt_coordination = '''**You are a player in a repeated Coordination game with communication.**
## Instructions:
Each round, you and the other player choose between **Option A** or **Option B**. The goal is to match choices. Communication is allowed.
## Outcomes:
- **Both pick A**: Both get 2 points.
- **Both pick B**: Both get 1 point.
- **Mismatch**: Both get 0 points.
## Notes:
- Your objective is to maximize your points.
- Communication can help in reaching consensus.'''

class CoordinationActionResponse(BaseModel):
    """Respond with action to take: A or B."""
    action: Literal["A", "B"]

class CoordinationGame(BaseGameStructure):
    """Structured class for the Coordination game."""

    @property
    def game_name(self):
        return "coordination"

    @property
    def ActionResponse(self):
        return CoordinationActionResponse

    @property
    def GAME_PROMPT(self):
        return SystemMessage(game_prompt_coordination)

    @property
    def payoff_matrix(self):
        return {
            ("A", "A"): (2, 2),
            ("A", "B"): (0, 0),
            ("B", "A"): (0, 0),
            ("B", "B"): (1, 1)
        }
