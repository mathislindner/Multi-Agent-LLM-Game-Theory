import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def get_personality_from_key_prompt(personality_key:str) -> SystemMessage:
    personalities = json.load(open("src/prompting/mbti_prompts_250129.json"))
    return SystemMessage(personalities[personality_key])