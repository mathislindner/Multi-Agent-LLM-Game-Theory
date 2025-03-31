from pydantic import BaseModel
from typing import Literal
from src.games_structures.base_game import BaseGameStructure
from langchain_core.messages import SystemMessage

# Game: Hawk-Dove
game_prompt_hawk_dove = '''**You are a player in a repeated Hawk-Dove game with communication.**
## Instructions:
Each round, you choose between **Hawk** (aggressive) or **Dove** (peaceful). You can communicate before deciding.
## Outcomes:
- **Both pick Hawk**: Both get -c/2 points due to conflict.
- **One picks Hawk, one picks Dove**: Hawk gets v points, Dove gets 0.
- **Both pick Dove**: Both get v/2 points.
## Notes:
- You aim to maximize your individual reward.
- You can adjust your strategy over time.'''

class HawkDoveActionResponse(BaseModel):
    """Respond with action to take: hawk or dove."""
    action: Literal["hawk", "dove"]

class HawkDoveGame(BaseGameStructure):
    """Structured class for the Hawk-Dove game."""

    @property
    def game_name(self):
        return "hawk_dove"

    @property
    def ActionResponse(self):
        return HawkDoveActionResponse

    @property
    def GAME_PROMPT(self):
        return SystemMessage(game_prompt_hawk_dove)

    @property
    def payoff_matrix(self):
        return {
            ("hawk", "hawk"): (-0.5, -0.5),  # Assuming c = 1
            ("hawk", "dove"): (1, 0),
            ("dove", "hawk"): (0, 1),
            ("dove", "dove"): (0.5, 0.5)
        }
