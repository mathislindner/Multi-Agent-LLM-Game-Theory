from models.test_model import TestModel

class baseAgent():
    name : str = "None"
    prompt : str = ""

    def __init__(self, name, prompt):
        self.name = name
        self.append_prompt(prompt)

    def append_prompt(self, message):
        self.prompt += message
        
def launch_game():
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

    call_for_message = "\n Please send a message to the other player. You can say 'cooperate' or 'defect'."

    call_for_decision = "\n Please make your decision. You can say 'cooperate' or 'defect'."

    agent1 = baseAgent("agent1", game_prompt)
    agent2 = baseAgent("agent2", game_prompt)

    agent1.append_prompt(agent_1_prompt)
    agent2.append_prompt(agent_2_prompt)

    agent1.append_prompt(altruist_agent_prompt)
    agent2.append_prompt(selfish_agent_prompt)

    agent1.append_prompt(call_for_message + "\n agent 1 (you): ")
    agent2.append_prompt(call_for_message + "\n agent 2 (you): ")

    agent1_message = llm.invoke(agent1.prompt)
    agent2_message = llm.invoke(agent2.prompt)

    def add_prompts_from_messages(agent1_message, agent2_message):
        reply_agent_1 = agent1_message.split("you): ")[-1]
        reply_agent_2 = agent2_message.split("you): ")[-1]
        agent1.append_prompt(reply_agent_2 + call_for_decision +"\n agent 1 (you): ")
        agent2.append_prompt(reply_agent_1 + call_for_decision +"\n agent 2 (you): ")

    add_prompts_from_messages(agent1_message, agent2_message)

    agent1_decision = llm.invoke(agent1.prompt).split("you): ")[-1]
    agent2_decision = llm.invoke(agent2.prompt).split("you): ")[-1]

    def evaluate_outcome(agent1_decision, agent2_decision):
        payoff_matrix = {
            ("cooperate", "cooperate"): (2, 2),
            ("cooperate", "defect"): (0, 3),
            ("defect", "cooperate"): (3, 0),
            ("defect", "defect"): (1, 1),
        }
        return payoff_matrix[(agent1_decision, agent2_decision)]

    outcome = evaluate_outcome(agent1_decision, agent2_decision)

    print(f"Agent 1 decided to {agent1_decision}")
    print(f"Agent 2 decided to {agent2_decision}")
    print(f"Outcome: {outcome}")

