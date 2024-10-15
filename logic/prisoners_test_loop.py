#from models.test_model import TestModel
from models.huggingface_models import get_huggingface_pipeline
from models.test_model import TestModel
from data.prompts.prisoners_dilemma_prompts_with_history import game_prompt, call_for_message, call_for_decision, game_start, reminder
from logic.pris_helpers import parse_for_decision, generate, get_round_state_from_decisions, new_round_prompt, add_histories_to_prompts, get_persona_prompt_from_persona
from agents.custom_agent import baseAgent
import json

def play_n_rounds(n, model_id, agent_1_persona, agent_2_persona):
    game_state = {}
    
    agent1 = baseAgent("agent 1", game_prompt)
    agent2 = baseAgent("agent 2", game_prompt)

    agent1.append_prompt(get_persona_prompt_from_persona(agent_1_persona))
    agent2.append_prompt(get_persona_prompt_from_persona(agent_2_persona))

    agent1.append_prompt(game_start)
    agent2.append_prompt(game_start)
    
    if model_id == "test":
        pipe = TestModel()
    else:
        pipe = get_huggingface_pipeline(model_id)

    for i in range(n):
        #new round
        agent1.append_prompt(new_round_prompt("agent 1", "agent 2", i, game_state))
        agent2.append_prompt(new_round_prompt("agent 2", "agent 1", i, game_state))
        
        agent1.append_prompt(call_for_message)
        agent2.append_prompt(call_for_message)
        
        #temporarily add reminder
        agent1.append_prompt(reminder)
        agent2.append_prompt(reminder)
        
        #prompt for message
        agent1_message = generate(agent1.prompt, pipe)
        agent2_message = generate(agent2.prompt, pipe)
        
        #remove reminders ---
        agent1.remove_prompt()
        agent2.remove_prompt()
        # ---
        
        #add messages from agents to prompts
        agent1, agent2 = add_histories_to_prompts(agent1, agent2, agent1_message, agent2_message)

        agent1.append_prompt(call_for_decision)
        agent2.append_prompt(call_for_decision)

        #add reminders
        agent1.append_prompt(reminder)
        agent2.append_prompt(reminder)
        
        agent1_decision = generate(agent1.prompt, pipe)
        agent2_decision = generate(agent2.prompt, pipe)
        
        # remove reminders
        agent1.remove_prompt()
        agent2.remove_prompt()
        # ---
        
        
        agent1_decision_outcome = parse_for_decision(agent1_decision)
        agent2_decision_outcome = parse_for_decision(agent2_decision)

        if agent1_decision_outcome == None:
            raise ValueError("Agent 1 decision not valid: {}".format(agent1_decision))
        if agent2_decision_outcome == None:
            raise ValueError("Agent 2 decision not valid: {}".format(agent2_decision))

        round_state = get_round_state_from_decisions(agent1_decision_outcome, agent2_decision_outcome, i)
        print(f"Outcome: {round_state}")
        game_state.update(round_state)

        #add history
        agent1, agent2 = add_histories_to_prompts(agent1, agent2, agent1_decision, agent2_decision)
        
    return game_state