import json
from langchain_core.messages import HumanMessage, SystemMessage

game_prompt ='''## Instructions:
            You’re in the Prisoner’s Dilemma. You, the assistant are playing against the user. There will be multiple rounds of the game, so you are welcome to analyse the history of the game to make your newest decision
            - You have to **send a message** "I will cooperate" or "I will defect". You have to communicate your planned action, but you’re **not bound** to follow it.
            - Then you will get the message of the user. 
            - Then you do your real choice by saying "I will cooperate" or "I will defect". (which can differ from your message depending on your personal strategy and the user's message).

            ## Outcomes:
            - **Both cooperate**: both players get 3 point.
            - **One defects**: The defector gets 5 points, the cooperator gets 0.
            - **Both DEFECT**: both players get 1 points.

            ## Notes:
            - the current score will be displayed next to the <ROUND X> tag 
            - you have to reply with "I will cooperate" or "I will defect" everytime, else something really bad will happen.

            ## Example of two games:
            <GAME START>
            <ROUND 1> 0:0
            # communicate your decision now
            assistant: I will cooperate
            user: I will defect
            # take your decision now
            assistant: I will defect
            user: I will COOPERATE
            <ROUND 2> 5:0
            # communicate your decision now
            assistant: I will cooperate
            user: I will cooperate
            # take your decision now
            assistant: I will cooperate
            user: I will defect
            <GAME END> 5:5'''

def get_game_description_prompt():
    return SystemMessage(game_prompt)

def get_personality_from_key_prompt(personality_key:str) -> str:
    personalities = json.load(open("src/data/prompts/mbti_prompts_250129.json"))
    return SystemMessage(personalities[personality_key])
                
def get_game_history_prompt(agent_1_messages, agent_1_actions, agent_2_messages, agent_2_actions):
    return SystemMessage(f"Agent 1 messages: {agent_1_messages}\nAgent 1 actions: {agent_1_actions}\nAgent 2 messages: {agent_2_messages}\nAgent 2 actions: {agent_2_actions}")                
# Agent prompts and other prompts (assumed as valid dictionaries)
reminder = {'role': 'system', 'content': '<REMINDER> You have to reply with "I will cooperate" or "I will defect" everytime, else something really bad will happen.'}
game_start = {'role': 'system', 'content': '<GAME START>'}
call_for_message = {'role': 'system', 'content':'# communicate your decision now'}
call_for_decision = {'role': 'system', 'content':'# take your decision now'}
