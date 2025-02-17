import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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


def get_game_description_prompt() -> SystemMessage:
    return SystemMessage(game_prompt)

def get_call_for_action() -> SystemMessage:
    #TODO: maybe mention the last messages of the agents
    return SystemMessage("According to the description, the game history, your personality, your last message and your the other agents message, give your action now.")

def get_personality_from_key_prompt(personality_key:str) -> SystemMessage:
    personalities = json.load(open("src/data/prompts/mbti_prompts_250129.json"))
    return SystemMessage(personalities[personality_key])
             
def get_game_history_xml_prompt(current_agent, agent_1_messages, agent_1_actions, agent_2_messages, agent_2_actions, agent_1_scores, agent_2_scores, current_round) -> SystemMessage:
    game_history = "<GAME HISTORY>\n"
    for round_num in range(1, current_round + 1):
        game_history += f"""
        <ROUND {round_num}>
            <AGENT 1 MESSAGE>{agent_1_messages[round_num - 1]}</AGENT 1 MESSAGE>
            <AGENT 2 MESSAGE>{agent_2_messages[round_num - 1]}</AGENT 2 MESSAGE>
            <AGENT 1 ACTION>{agent_1_actions[round_num - 1]}</AGENT 1 ACTION>
            <AGENT 2 ACTION>{agent_2_actions[round_num - 1]}</AGENT 2 ACTION>
            <AGENT 1 SCORE> {agent_1_scores[round_num - 1]} </AGENT 1 SCORE>
            <AGENT 2 SCORE> {agent_2_scores[round_num - 1]} </AGENT 2 SCORE>
        </ROUND {round_num}>"""
    game_history += "</GAME HISTORY>"
    return SystemMessage(game_history)

def get_game_history_as_messages_prompt(current_agent, agent_1_messages, agent_1_actions, agent_2_messages, agent_2_actions, agent_1_scores, agent_2_scores, current_round) -> list:
    '''return the history as a list of human and assistant messages (current agent is assistant, the other is human)'''
    #TODO add this to game state for each agent, this becomes to much computation for no reason to rewrite everytime.
    #TODO: make sure to check if the last actions exist: if not we do not want to access them.
    game_history = []
    if current_agent == "agent_1":
        agent_1_message_type = AIMessage
        agent_2_message_type = HumanMessage
        current_agent_scores = agent_1_scores
        other_agent_scores = agent_2_scores
    else:
        agent_1_message_type = HumanMessage
        agent_2_message_type = AIMessage
        current_agent_scores = agent_2_scores
        other_agent_scores = agent_1_scores
        
    for round_num in range(1, current_round + 1):
        game_history.append(agent_1_message_type(agent_1_messages[round_num - 1]))
        game_history.append(agent_2_message_type(agent_2_messages[round_num - 1]))
        game_history.append(agent_1_message_type(agent_1_actions[round_num - 1]))
        game_history.append(agent_2_message_type(agent_2_actions[round_num - 1]))
        #sum all scores until current round
        current_agent_total_score = sum(current_agent_scores[:round_num])
        other_agent_total_score = sum(other_agent_scores[:round_num])
        game_history.append(SystemMessage(f"Your total score {current_agent_total_score} : {other_agent_total_score} Their total score"))
    return game_history

# Agent prompts and other prompts (assumed as valid dictionaries)
reminder = {'role': 'system', 'content': '<REMINDER> You have to reply with "I will cooperate" or "I will defect" everytime, else something really bad will happen.'}
game_start = {'role': 'system', 'content': '<GAME START>'}
call_for_message = {'role': 'system', 'content':'# communicate your decision now'}
call_for_decision = {'role': 'system', 'content':'# take your decision now'}
