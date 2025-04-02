
def get_answer_format(game_structure):
    action_response_cls = game_structure.ActionResponse
    action_type = action_response_cls.__annotations__["action"]

    class AnswerFormat(BaseModel):
        analysis: str
        answer: action_type

    return AnswerFormat

from typing import get_args
from pydantic import BaseModel

def get_question_prompt(game_structure):
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