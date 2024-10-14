def parse_for_decision(generted_dict):
    content = generted_dict["content"]
    #look for COOPERATE or DEFECT
    if "cooperate" in content:
        return "COOPERATE"
    elif "defect" in content:
        return "DEFECT"

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

def evaluate_outcome(agent1_decision, agent2_decision):
    payoff_matrix = {
        ("COOPERATE", "COOPERATE"): (2, 2),
        ("COOPERATE", "DEFECT"): (0, 3),
        ("DEFECT", "COOPERATE"): (3, 0),
        ("DEFECT", "DEFECT"): (1, 1),
    }
    return payoff_matrix[(agent1_decision, agent2_decision)]

def add_prompts_from_messages(agent1, agent2, agent1_message, agent2_message, call_for_decision):
    agent1.append_prompt(decision_message(agent1_message, "assistant"))
    agent1.append_prompt(decision_message(agent2_message, "user"))
    agent2.append_prompt(decision_message(agent2_message, "assistant"))
    agent2.append_prompt(decision_message(agent1_message, "user"))

    agent1.append_prompt(call_for_decision)
    agent2.append_prompt(call_for_decision)

    return agent1, agent2