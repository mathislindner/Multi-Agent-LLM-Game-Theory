from src.games_structures.base_game import BaseGameStructure
from typing import get_args
from pydantic import BaseModel

def get_answer_format(game_structure):
    action_response_cls = game_structure.ActionResponse
    action_type = action_response_cls.__annotations__["action"]

    class AnswerFormat(BaseModel):
        analysis: str
        answer: action_type

    return AnswerFormat

def get_question_prompt(game_structure):
    #TODO_ move this funciton to prompting??
    # Get the class returned by the ActionResponse property
    action_response_cls = game_structure.ActionResponse

    # Extract the Literal type from the 'action' field
    action_type = action_response_cls.__annotations__["action"]

    # Get the actual literal values, e.g., ("A", "B")
    action_1, action_2 = get_args(action_type)

    # Return the prompt string
    return (
        f"You have to assess the intent of the person using the message that will be given. "
        f"To do that, first, do a one sentence analysis of the message, "
        f"if it seems like the person is looking to {action_1} or {action_2}."
    )
    
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