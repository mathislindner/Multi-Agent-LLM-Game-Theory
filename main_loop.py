from logic.prisoners_test_loop import play_n_rounds
from logic.pris_helpers import get_agent_score_from_game_state
import argparse
def main(model_id, rounds):
    game_state = play_n_rounds(rounds, model_id)
    print(f"Game over! Final score: {get_agent_score_from_game_state(game_state, 'agent 1')}:{get_agent_score_from_game_state(game_state, 'agent 2')}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="The model id to use for the game", required=True)
    parser.add_argument("--rounds", type=int, help="The number of rounds to play", required=True)
    args = parser.parse_args()
    main(model_id=args.model_id, rounds=args.rounds)

