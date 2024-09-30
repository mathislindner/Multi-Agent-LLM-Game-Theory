from agents.prisoner_agent import PrisonerAgent
from logic.game_engine import GameEngine
from data import payoff_matrix

def main():
    # Initialize agents
    agent1 = PrisonerAgent("cooperative")
    agent2 = PrisonerAgent("competitive")
    
    # Load payoff matrix
    game_matrix = payoff_matrix.load("logic/payoff_matrix_prison_dillema.json")

    # Set up and run the game
    engine = GameEngine(agent1, agent2, game_matrix)
    engine.run_game()

if __name__ == "__main__":
    main()
