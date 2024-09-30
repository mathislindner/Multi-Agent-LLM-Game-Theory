class GameEngine:
    def __init__(self, agent1, agent2, payoff_matrix):
        self.agent1 = agent1
        self.agent2 = agent2
        self.payoff_matrix = payoff_matrix

    def run_game(self):
        # Simulate the game, coordinating between agents
        agent1_message = self.agent1.communicate()
        agent2_message = self.agent2.communicate()
        
        agent1_decision = self.agent1.decide(agent2_message)
        agent2_decision = self.agent2.decide(agent1_message)
        
        outcome = (agent1_decision, agent2_decision)
        self.evaluate_outcome(outcome)

    def evaluate_outcome(self, outcome):
        payoffs = self.payoff_matrix[outcome]
        print(f"Outcome: {outcome}, Payoffs: {payoffs}")
