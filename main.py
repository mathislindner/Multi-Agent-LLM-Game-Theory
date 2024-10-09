#from logic.prisoners_dilemma_intent_game_engine import run_game
from logic.prisoners_test import launch_game
#from logic.prisoners_test_parallel import launch_game
#from models.huggingface_models import try_huggingface_model
import argparse
def main(model_id):
    launch_game(model_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="The model id to use for the game", required=True)
    args = parser.parse_args()
    main(model_id=args.model_id)

