
class baseAgent():
    def __init__(self, name, prompt):
        self.name = name
        self.prompt = [prompt]

    def append_prompt(self, message):
        self.prompt.append(message)

    def remove_prompt(self):
        return self.prompt.pop()

import random    
class randomAgent():
    def __init__(self):
        self.name = "random agent"
        self.prompt = []

    def append_prompt(self, message):
        self.prompt.append(message)