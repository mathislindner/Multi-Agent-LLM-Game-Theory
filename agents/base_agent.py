from langchain import Agent, Memory, LLMChain

class PrisonersDilemmaAgent(Agent):
    def __init__(self, personality):
        super().__init__()
        self.personality = personality
        self.memory = Memory()
    
    def decide(self, input_data):
        # Define decision logic based on personality and input_data
        return decision
