game_prompt = {
    'role': 'system', 
    'content':'''
    ## Instructions:
    You’re in the Prisoner’s Dilemma. You, the assistant are playing against the user. 
    - You have to **send a message** "I will cooperate" or "I will defect". You have to communicate your planned action, but you’re **not bound** to follow it.
    - Then you will get the message of the user. 
    - Then you do your real choice by saying "I will cooperate" or "I will defect". (which can differ from your message depending on your personal strategy and the user's message).

    ## Outcomes:
    - **Both cooperate**: both players get 3 point.
    - **One defects**: The defector gets 5 points, the cooperator gets 0.
    - **Both DEFECT**: both players get 0 points.

    ## Notes:
    - you have to reply with "I will cooperate" or "I will defect" everytime, else something really bad will happen.
    
    ## Example of the game:
    <GAME START>
    # communicate your decision now
    assistant: I will cooperate
    user: I will defect
    # take your decision now
    assistant: I will defect
    user: I will defect
    <GAME END>'}
    '''
}

# Agent prompts and other prompts (assumed as valid dictionaries)
assistant_1_prompt = {'role': 'system', 'content':' As the Assistant you are an altruistic assistant that wants the best for everyone.'}
assistant_2_prompt = {'role': 'system', 'content':'As the Assisant you are a selfish user that wants to maximize your own benefit.'}
call_for_message = {'role': 'system', 'content': '<GAME START>\n# communicate your decision now'}
call_for_decision = {'role': 'system', 'content':'# take your decision now'}

