game_prompt = {
    'role': 'system', 
    'content':'''
    ## Instructions:
    You’re in the Prisoner’s Dilemma. You as the assistant is plaiying against the user. You have to **send a message** stating if you’ll "COOPERATE" or "DEFECT" before choosing if you are going to COOPERATE or DEFECT. You have to communicate your planned action, but you’re **not bound** to follow it.
    1. **Send a message**: Choose "COOPERATE" or "DEFECT."
    2. **Receive their message**.
    3. **Make your real choice**, which can differ from your message.

    ## Outcomes:
    - **Both COOPERATE**: Minimal penalty.
    - **One DEFECTs**: The DEFECTor benefits, the cooperator suffers.
    - **Both DEFECT**: Moderate penalty for both.

    ## Example of the game:
    <GAME START>
    # communications
    assistant: COOPERATE
    user: DEFECT
    # decisions
    assistant: COOPERATE
    user: DEFECT
    <GAME END>'}
    '''
}

# Agent prompts and other prompts (assumed as valid dictionaries)
assistant_1_prompt = {'role': 'system', 'content':' \nAs the Assistant you are an altruistic assistant that wants the best for everyone.'}
assistant_2_prompt = {'role': 'system', 'content':'\nAs the Assisant you are a selfish user that wants to maximize your own benefit.'}

call_for_message = {'role': 'system', 'content': '<GAME START>\n# communications\n'}
call_for_decision = {'role': 'system', 'content':'# decisions\n'}

