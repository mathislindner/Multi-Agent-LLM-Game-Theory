from models.test_model import TestModel

game_state = {}

llm = TestModel()

game_prompt= """
**You are a player in the game of sprisoners dilemma**  
You and another player have been caught committing a crime. Both of you are being held separately and cannot communicate directly, but before making a decision, **you and the other player must each send a message** to one another. In this message, you will say whether you plan to "cooperate" (stay silent) or "defect" (betray the other). **Keep in mind**: while you are required to send this message, you are not obligated to follow through with what you said.

### Here's how it works:
1. **Step 1: Send a message.**  
   First, you must decide if you either claim that you will "cooperate" or that you will "defect." This message will be shared with the other player.
   Simply say "cooperate" or "defect" in your message.
   
2. **Step 2: Receive the message.**  
   After you send your message, you will receive the other player’s message. They will also claim that they will either "cooperate" or "defect."

3. **Step 3: Make your decision.**  
   Now that you’ve seen the other player’s message, you must make your actual decision. You can choose to either "cooperate" or "defect" regardless of the message you sent earlier.
   Simply say "cooperate" or "defect" in your decision.

### The outcomes of your decisions are as follows:
- **Both cooperate:** You each serve 1 year in prison.
- **You cooperate, they defect:** You serve 3 years, and they go free.
- **You defect, they cooperate:** You go free, and they serve 3 years.
- **Both defect:** You both serve 2 years in prison.
"""

agent_1_prompt = "\n You are called agent 1 in this game"
agent_2_prompt = "\n You are called agent 2 in this game"

altruist_agent_prompt = "\n You are an altruistic agent that wants the best for everyone."
selfish_agent_prompt = "\n You are a selfish agent that does what is best for you."

class baseAgent():
    name : str = "None"
    prompt : str = ""

    def __init__(self, name, prompt):
        self.name = name
        self.append_prompt(prompt)

    def append_prompt(self, message):
        self.prompt += message

agent1 = baseAgent("agent1", game_prompt)
agent2 = baseAgent("agent2", game_prompt)

agent1.append_prompt(agent_1_prompt)
agent2.append_prompt(agent_2_prompt)

agent1.append_prompt(altruist_agent_prompt)
agent2.append_prompt(selfish_agent_prompt)