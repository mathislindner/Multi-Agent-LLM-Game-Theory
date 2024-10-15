from logic.prisoners_test_loop import play_n_rounds
from logic.pris_helpers import get_agent_score_from_game_state
from logic.plotting import create_plots
import argparse
import json
import os

def main(model_id, rounds, agent_1_persona, agent_2_persona):
    out_file_name = f"outputs/{agent_1_persona}_{agent_2_persona}_{rounds}"
    game_state = play_n_rounds(rounds, model_id, agent_1_persona, agent_2_persona)
    print(f"Game over! Final score: {get_agent_score_from_game_state(game_state, 'agent 1')}:{get_agent_score_from_game_state(game_state, 'agent 2')}")
    
    #save game state to file and plot
    
    with open(f"{out_file_name}.json", "w") as f:
        f.write(json.dumps(game_state, indent=4))
    create_plots(game_state, agent_1_persona, agent_2_persona,int(rounds), out_file_name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="The model id to use for the game", required=True)
    parser.add_argument("--rounds", type=int, help="The number of rounds to play", required=True)
    parser.add_argument("--agent_1_persona", choices=['selfish', 'altruistic','no_persona'], help="The personality of agent 1", required=True)
    parser.add_argument("--agent_2_persona", choices=['selfish', 'altruistic','no_persona'], help="The personality of agent 2", required=True)
    args = parser.parse_args()
    main(model_id=args.model_id, rounds=args.rounds, agent_1_persona=args.agent_1_persona, agent_2_persona=args.agent_2_persona)

