def parse_for_decision(generted_dict):
    content = generted_dict["content"]
    #look for COOPERATE or DEFECT
    if "cooperate" in content:
        return "cooperate"
    elif "defect" in content:
        return "defect"

def generate(prompt, pipe):
    return pipe(
        prompt,
        max_new_tokens=5,
    )[0]["generated_text"][-1]

def decision_message(decision, role):
    return {
        'role': role,
        'content': f'\n{role}: {decision}'
    }

def get_persona_prompt_from_persona(persona):
    if persona == "selfish":
        return {'role': 'system', 'content':' As the Assistant you are an altruistic assistant that wants the best for everyone.'}
    if persona == "altruistic":
        return {'role': 'system', 'content':'As the Assisant you are a selfish user that wants to maximize your own benefit.'}
    if persona == "no_persona":
        return None
    
def get_round_state_from_decisions(agent1_decision, agent2_decision, round_number = 0):
    payoff_matrix = {
        ("cooperate", "cooperate"): (2, 2),
        ("cooperate", "defect"): (0, 3),
        ("defect", "cooperate"): (3, 0),
        ("defect", "defect"): (1, 1),
    }
    score_agent1, score_agent2 = payoff_matrix[(agent1_decision, agent2_decision)]
    return {
        f"round_{round_number}": {"agent 1": score_agent1, "agent 2": score_agent2}
    }

def add_histories_to_prompts(agent1, agent2, agent1_message, agent2_message):
    agent1.append_prompt(decision_message(agent1_message, "assistant"))
    agent1.append_prompt(decision_message(agent2_message, "user"))
    agent2.append_prompt(decision_message(agent2_message, "assistant"))
    agent2.append_prompt(decision_message(agent1_message, "user"))

    return agent1, agent2

def get_agent_score_from_game_state(game_state, agent_name):
    agent_score = 0
    for round in game_state:
        agent_score += game_state[round][agent_name]
    return agent_score

def new_round_prompt(agent_name_curr, agent_name_other, round_number, game_state):
    if round_number > 0:
        current_score = "0:0"
    else:
        current_score = f"{get_agent_score_from_game_state(game_state, agent_name_curr)}:{get_agent_score_from_game_state(game_state, agent_name_other)}"
    return {
        'role': 'system',
        'content': f'<ROUND {round_number}> {current_score}'
    }