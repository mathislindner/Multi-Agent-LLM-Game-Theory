from logic.prisoners_dilemma_intent_game_engine import GameEngine
import argparse
def main():
    game = GameEngine()
    game.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Run the game", action="store_true")
    args = parser.parse_args()
    main()

