#from logic.prisoners_dilemma_intent_game_engine import run_game
from logic.prisoners_test import launch_game
#from models.huggingface_models import try_huggingface_model
import argparse
def main():
    if args.run:
        #run_game()
        launch_game()
        #try_huggingface_model()
    else:
        print("Please provide the --run flag to run the game")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Run the game", action="store_true")
    args = parser.parse_args()
    main()

