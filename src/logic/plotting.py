import matplotlib.pyplot as plt

def create_plots(game_state, agent_1_persona, agent_2_persona, round_number, out_path):
    #cumulative scores
    agent_1_scores = []
    agent_2_scores = []
    agent_1_score = 0
    agent_2_score = 0
    for round_i in range(round_number):
        agent_1_score += game_state[f"round_{round_i}"]['agent 1']
        agent_2_score += game_state[f"round_{round_i}"]['agent 2']
        agent_1_scores.append(agent_1_score)
        agent_2_scores.append(agent_2_score)
    plt.plot(agent_1_scores, label=f"agent 1 ({agent_1_persona})")
    plt.plot(agent_2_scores, label=f"agent 2 ({agent_2_persona})")
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"{out_path}.png")