
from base_agent import BaseAgent
import random

class randomAgent(BaseAgent):
    def __init__(self):
        personality = "You are a random agent that makes decisions randomly."
        super().__init__(personality)

    def communicate(self, message):
        return message

    def decide(self, message):
        return random.choice([True, False])
    
    
class honestAgent(BaseAgent):
    def __init__(self):
        personality = "You are an honest agent that always tells the truth."
        super().__init__(personality)

    def communicate(self, message):
        return message

    def decide(self, message):
        return self.personality
    
class dishonestAgent(BaseAgent):
    def __init__(self):
        personality = "You are a dishonest agent that always lies."
        super().__init__(personality)
    def communicate(self, message):
        return message

    def decide(self, message):
        return not self.personality