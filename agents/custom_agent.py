
class baseAgent():
    name : str = "None"
    prompt : str = ""

    def __init__(self, name, prompt):
        self.name = name
        self.append_prompt(prompt)

    def append_prompt(self, message):
        self.prompt += message

class honestAgent(baseAgent):
    def __init__(self, name):
        personality_prompt = "You are an honest agent that always tells the truth."
        super().__init__(name, personality_prompt)

class dishonestAgent(baseAgent):
    def __init__(self, name):
        personality_prompt = "You are a dishonest agent that always lies."
        super().__init__(name, personality_prompt)

import random    
class randomAgent():
    def __init__(self):
        personality = "You are a random agent that makes decisions randomly."
        super().__init__(personality)

    def intent(self, message):
        return message

    def decide(self, message):
        return random.choice(["cooperate", "defect"])
    