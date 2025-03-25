import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

#TODO: figure out how we want to personality and history in the context of our classes and function
def get_personality_from_key_prompt(personality_key:str) -> SystemMessage:
    personalities = json.load(open("src/prompting/mbti_prompts_250129.json"))
    return SystemMessage(personalities[personality_key])

def get_game_history_as_messages_prompt(current_agent, state, history_type: str):
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
    return game_history