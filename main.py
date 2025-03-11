from src.run_prisoners_mbti import run_n_rounds_w_com
from src.judge_messages import judge_message_intent
from datetime import datetime
#from logic.plotting import create_plots
import argparse
import json
import os

def main(model_id, rounds, agent_1_persona, agent_2_persona, game_name):
    date_string = datetime.now().strftime("%y%m%d")
    output_dir = "src/data/outputs/"
    base_game_state_path = f"{output_dir}{game_name}_{date_string}"
    game_state_path = base_game_state_path
    n = 0
    while os.path.exists(game_state_path):
        game_state_path = f"{base_game_state_path}_{n}.csv"
        n += 1
    judged_game_state_path = f"{game_state_path[:-4]}_judged.csv"
    game_state = run_n_rounds_w_com(model_name = model_id, total_rounds = rounds, personality_key_1 = agent_1_persona, personality_key_2 = agent_2_persona, game_name = game_name, file_path = game_state_path)
    judge_message_intent(file_path_in = game_state_path, file_path_out= judged_game_state_path, model_name = model_id)
    
if __name__ == "__main__":
    #main(model_id= 'gpt-4o-mini', rounds=3, agent_1_persona="INFP", agent_2_persona = "ENTP")
    personality_choices = json.load(open("src/prompting/mbti_prompts_250129.json")).keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="The model id to use for the game", required=True)
    parser.add_argument("--rounds", type=int, help="The number of rounds to play", required=True)
    parser.add_argument("--agent_1_persona", choices = personality_choices, help="The personality of agent 1", required=True)
    parser.add_argument("--agent_2_persona", choices = personality_choices, help="The personality of agent 2", required=True)
    parser.add_argument("--game_name", choices = ["prisoners_dilemma", "stag_hunt"], help="The game to play")
    args = parser.parse_args()
    main(model_id=args.model_id, rounds=args.rounds, agent_1_persona=args.agent_1_persona, agent_2_persona=args.agent_2_persona, game_name=args.game_name)

