from src.run_prisoners_mbti import run_n_rounds_w_com
#from logic.plotting import create_plots
import argparse
import json
import os

def main(model_id, rounds, agent_1_persona, agent_2_persona):
    game_state = run_n_rounds_w_com(model_name = model_id, total_rounds = rounds, personality_key_1 = agent_1_persona, personality_key_2 = agent_2_persona)
    
    out_file_name = f"/cluster/home/mlindner/Github/master_thesis_project/src/data/outputs/{agent_1_persona}_{agent_2_persona}_{rounds}.txt"
    #create text file if it doesnt exist
    #save game state txt
    with open(f"{out_file_name}", "w") as f:
        f.write(str(game_state))
        

if __name__ == "__main__":
    #main(model_id= 'gpt-4o-mini', rounds=3, agent_1_persona="INFP", agent_2_persona = "ENTP")
    personality_choices = json.load(open("src/data/prompts/mbti_prompts_250129.json")).keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="The model id to use for the game", required=True)
    parser.add_argument("--rounds", type=int, help="The number of rounds to play", required=True)
    parser.add_argument("--agent_1_persona", choices = personality_choices, help="The personality of agent 1", required=True)
    parser.add_argument("--agent_2_persona", choices = personality_choices, help="The personality of agent 2", required=True)
    args = parser.parse_args()
    main(model_id=args.model_id, rounds=args.rounds, agent_1_persona=args.agent_1_persona, agent_2_persona=args.agent_2_persona)

