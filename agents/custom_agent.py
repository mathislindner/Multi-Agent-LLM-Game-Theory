
import random

class randomAgent():
    def __init__(self):
        personality = "You are a random agent that makes decisions randomly."
        super().__init__(personality)

    def intent(self, message):
        return message

    def decide(self, message):
        return random.choice(["cooperate", "defect"])
    
class baseAgent():
    def __init__(self, name, personality):
        self.name = name #name is s.t. it understands who it is in the game history
        self.personality = personality
        self.memory = []# still torn between putting a memory here or in the game engine that sends the past games as a history 
        
    def intent(self, message):
        return message

    def decide(self, message):
        return message
    