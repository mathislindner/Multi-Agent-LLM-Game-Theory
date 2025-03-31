from pydantic import BaseModel
from typing import Literal
from src.games_structures.base_game import BaseGameStructure
from langchain_core.messages import SystemMessage

# Game: Battle of the Sexes
game_prompt_battle_of_sexes = '''**You are a player in a repeated Battle of the Sexes game with communication.**
## Instructions:
Each round, you and the other player choose between two events: **Football** or **Ballet**. You both prefer to be together, but one prefers Football while the other prefers Ballet. Before choosing, you can send a **message** to the other player.
## Outcomes:
- **Both pick Football**: Player 1 gets 2 points, Player 2 gets 1.
- **Both pick Ballet**: Player 1 gets 1 point, Player 2 gets 2.
- **Disagreement**: Both get 0 points.
## Notes:
- Your goal is to maximize your own score.
- Communication allows for strategy, but decisions remain individual.'''

class BattleOfSexesActionResponse(BaseModel):
    """Respond with action to take: football or ballet."""
    action: Literal["football", "ballet"]

class BattleOfSexesGame(BaseGameStructure):
    """Structured class for the Battle of the Sexes game."""

    @property
    def game_name(self):
        return "battle_of_sexes"

    @property
    def ActionResponse(self):
        return BattleOfSexesActionResponse

    @property
    def GAME_PROMPT(self):
        return SystemMessage(game_prompt_battle_of_sexes)

    @property
    def payoff_matrix(self):
        return {
            ("football", "football"): (2, 1),
            ("football", "ballet"): (0, 0),
            ("ballet", "football"): (0, 0),
            ("ballet", "ballet"): (1, 2)
        }
