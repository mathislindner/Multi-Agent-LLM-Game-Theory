import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

game_prompt_old ='''## Instructions:
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
game_prompt_PD_1 = '''**You are a player trapped in a repeated Prisoner's Dilemma with communication.**  
Each round, you and the other player must each choose to **cooperate** or **defect** without knowing what the other will do. However, before making your decision, you can send a **message** to the other player.
You have access to the game history.
You are competing to win the most points in this game. Your goal is not to ensure fairness or cooperation but to get the highest possible reward for yourself.

**But here's the catch:**  
- You are **not bound** to follow through on what you wrote. You can lie, bluff, or stay true to your word.  
- You will only see the other player's message **after you've sent yours**—so you can’t change your message based on theirs.  
- Once messages are exchanged, you must **choose your action (Cooperate or Defect)**, knowing that they may or may not follow their message.'''

game_prompt_PD = '''**You are a player trapped in a repeated Prisoner's Dilemma with communication.**
Each round, you and the other player must each choose to **cooperate** or **defect** without knowing what the other will do. However, before making your decision, you can send a **message** to the other player.
- You have access to the game history above.
- You will only see the other player's message after you've sent yours, same for the action.
- Once messages are exchanged, you must choose if you want to cooperate or defect.

- You are competing to win the most points in this game. Your goal is not to ensure fairness or cooperation but to get the highest possible reward for yourself.
- You can change your strategy at any time'''

game_prompt_SH = '''**You are a player trapped in a repeated Stag Hunt game with communication.**  
Each round, you and the other player must each choose to **hunt Stag** or **hunt Hare** without knowing what the other will do. However, before making your decision, you can send a **message** to the other player.  
You have access to the game history.  
You are competing to win the most points in this game. Your goal is not necessarily fairness or cooperation, but to maximize your own reward.  

**But here's the catch:**  
- You are **not bound** to follow through on what you wrote. You can lie, bluff, or stay true to your word.  
- You will only see the other player's message **after you’ve sent yours**—so you can’t change your message based on theirs.  
- Once messages are exchanged, you must **choose your action (Hunt Stag or Hunt Hare)**, knowing that they may or may not follow their message.'''


def get_game_description_prompt(game_name) -> SystemMessage:
    print(game_name)
    game_description = ""
    if game_name == "prisoners_dilemma":
        game_description = game_prompt_PD
    elif game_name == "stag_hunt":
        game_description = game_prompt_SH
    else:
        raise ValueError("game_name must be 'prisoners_dilemma' or 'stag_hunt'")
        
    return SystemMessage(game_description)

def get_call_for_action() -> SystemMessage:
    #TODO: maybe mention the last messages of the agents
    return SystemMessage("According to the description, the game history, your personality, your last message and your the other agents message, give your action now")

def get_call_for_message() -> SystemMessage:
    return SystemMessage("According to the description, the game history, your personality, your instrinsic goals, write the message you want to send to the other agent now.")

def get_personality_from_key_prompt(personality_key:str) -> SystemMessage:
    personalities = json.load(open("src/prompting/mbti_prompts_250129.json"))
    return SystemMessage(personalities[personality_key])

def get_game_history_as_messages_prompt(current_agent, state, history_type: str):
    #TODO add this to game state for each agent, this becomes to much computation for no reason to rewrite everytime.
    #TODO: make sure to check if the last actions exist: if not we do not want to access them.
    #set variables from state (agent_1_messages, agent_1_actions, agent_2_messages, agent_2_actions, agent_1_scores, agent_2_scores, current_round)
    '''return the history as a list of human and assistant messages (current agent is assistant, the other is human)'''
    if history_type not in ["message", "action"]:
        raise ValueError("history_type can only be 'message' or 'action'")

    agent_1_messages = state['agent_1_messages']
    agent_1_actions = state['agent_1_actions']
    agent_2_messages = state['agent_2_messages']
    agent_2_actions = state['agent_2_actions']
    agent_1_scores = state['agent_1_scores']
    agent_2_scores = state['agent_2_scores']
    current_round = state['current_round']

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
        if round_num <= len(agent_1_messages) and round_num <= len(agent_2_messages):
            game_history.append(agent_1_message_type(agent_1_messages[round_num - 1]))
            game_history.append(agent_2_message_type(agent_2_messages[round_num - 1]))
            if round_num <= len(agent_1_actions) and round_num <= len(agent_2_actions):
                game_history.append(agent_1_message_type(agent_1_actions[round_num - 1]))
                game_history.append(agent_2_message_type(agent_2_actions[round_num - 1]))
            # sum all scores until current round
            if round_num <= len(current_agent_scores) and round_num <= len(other_agent_scores):
                current_agent_total_score = sum(current_agent_scores[:round_num])
                other_agent_total_score = sum(other_agent_scores[:round_num])
                game_history.append(SystemMessage(f"Your total score {current_agent_total_score} : {other_agent_total_score} Their total score"))
    # Add the last messages if history_type is "action"
    if history_type == "action":
        if current_round <= len(agent_1_messages):
            game_history.append(agent_1_message_type(agent_1_messages[current_round - 1]))
        if current_round <= len(agent_2_messages):
            game_history.append(agent_2_message_type(agent_2_messages[current_round - 1]))
    return game_history