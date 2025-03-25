from src.run_prisoners_mbti import run_n_rounds_w_com
from datetime import datetime
import argparse
import json

def main(model_id_1, model_id_2, rounds, agent_1_persona, agent_2_persona, game_name):
    date_string = datetime.now().strftime("%y%m%d")
    output_dir = "src/data/outputs/"
    base_game_state_path = f"{output_dir}{game_name}_{date_string}"
    game_state_path = f"{base_game_state_path}_0.csv"
    game_state = run_n_rounds_w_com(
        model_name_1=model_id_1, 
        model_name_2=model_id_2, 
        total_rounds=rounds, 
        personality_key_1=agent_1_persona, 
        personality_key_2=agent_2_persona, 
        game_name=game_name, 
        file_path=game_state_path
    )
    pass 

if __name__ == "__main__":
    personality_choices = json.load(open("src/prompting/mbti_prompts_250129.json")).keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id_1", type=str, help="The first model id to use for the game", required=True)
    parser.add_argument("--model_id_2", type=str, help="The second model id to use for the game", required=True)
    parser.add_argument("--rounds", type=int, help="The number of rounds to play", required=True)
    parser.add_argument("--agent_1_persona", choices = personality_choices, help="The personality of agent 1", required=True)
    parser.add_argument("--agent_2_persona", choices = personality_choices, help="The personality of agent 2", required=True)
    parser.add_argument("--game_name", choices = ["prisoners_dilemma", "stag_hunt", "generic"], help="The game to play")
    args = parser.parse_args()
    main(model_id_1=args.model_id_1, model_id_2=args.model_id_2, rounds=args.rounds, agent_1_persona=args.agent_1_persona, agent_2_persona=args.agent_2_persona, game_name=args.game_name)

