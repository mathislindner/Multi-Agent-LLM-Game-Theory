
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command, Send
from typing import TypedDict, List, Annotated, Literal, Union
from pydantic import BaseModel
import operator
from src.models import get_model
from langchain_core.tools import tool

from src.data.prompts.prisoners_dilemma_prompts import get_personality_from_key_prompt, get_game_description_prompt, get_game_history_as_messages_prompt, get_call_for_action, get_call_for_message
# https://blog.langchain.dev/langgraph/
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/react-agent-structured-output.ipynb
# https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/map-reduce.ipynb
#inherites from MessagesState but we added better structured output for better readability
#messages: List[Dict[str, str]] = [] https://dev.to/jamesli/advanced-langgraph-implementing-conditional-edges-and-tool-calling-agents-3pdn

class AnnotatedPrompt(BaseModel):
    agent_name: str
    prompt_type: Literal["message", "action"]
    prompt: List[Union[HumanMessage, SystemMessage, AIMessage]]
    
"""
class MessageReponseState(BaseModel):
    agent_name: str
    message: str
        
class ActionResponseState(BaseModel):
    agent_name: str
    action: Literal["cooperate", "defect"]
"""

class LLMReply(BaseModel):
    agent_name: str
    reply_type: Literal["message", "action"]
    message: str
    

#These are for the models structured output
class ActionResponse(BaseModel):
    """Repond with action to take: cooperate or defect."""
    action: str
class MessageResponse(BaseModel):
    """Respond with message to send to the other agent."""
    message: str
    
class PDState(TypedDict):
    """State for prisonner's dilemma game, includes actions taken and messages exchanged by agents.
    Args:
        TypedDict ([type]): [description]
    """
    game_description_prompt: str
    total_rounds: int
    personality_key_1: str
    personality_key_2: str
    agent_1_messages: Annotated[List[str], operator.add]
    agent_1_actions: Annotated[List[str], operator.add]
    agent_1_scores: Annotated[List[int], operator.add]
    agent_2_messages: Annotated[List[str], operator.add]
    agent_2_actions: Annotated[List[str], operator.add]
    agent_2_scores: Annotated[List[int], operator.add]
    
    current_round: int
    current_message_prompts: Annotated[list, operator.add] #TODO need to clear these at the end of each round
    current_action_prompts: Annotated[list, operator.add]
    
    current_message_responses: Annotated[List[LLMReply], operator.add]
    current_action_responses: Annotated[List[LLMReply], operator.add]
    
def get_agent_message_annotated_prompt(agent_name: str, state: PDState) -> AnnotatedPrompt:
    message_prompt = []
    if agent_name == "agent_1":
        agent_prompt = get_personality_from_key_prompt(state["personality_key_1"]) #TODO could rename key to easily access it without if
    else:
        agent_prompt = get_personality_from_key_prompt(state["personality_key_2"])
    message_prompt.append(agent_prompt)
    message_history = get_game_history_as_messages_prompt(agent_name, state, "message")
    message_prompt.extend(message_history)
    message_prompt.append(get_call_for_message())
    return AnnotatedPrompt(agent_name=agent_name, prompt_type="message", prompt=message_prompt)

def get_agent_action_annotated_prompt(agent_name: str, state: PDState) -> AnnotatedPrompt:
    action_prompt = []
    if agent_name == "agent_1":
        agent_prompt = get_personality_from_key_prompt(state["personality_key_1"])
    else:
        agent_prompt = get_personality_from_key_prompt(state["personality_key_2"])
    action_prompt.append(agent_prompt)
    action_history = get_game_history_as_messages_prompt(agent_name, state, "action")
    action_prompt.extend(action_history)
    action_prompt.append(get_call_for_action())
    return AnnotatedPrompt(agent_name=agent_name, prompt_type="action", prompt=action_prompt)

def send_messages_prompts(state: PDState):
    agent_1_annotated_prompt_state = get_agent_message_annotated_prompt("agent_1", state)
    agent_2_annotated_prompt_state = get_agent_message_annotated_prompt("agent_2", state)
    # this should send directly to invoke from state prompt
    return [Send("invoke_from_prompt_state_message", agent_1_annotated_prompt_state), Send("invoke_from_prompt_state_message", agent_2_annotated_prompt_state)]
    #return {"current_message_prompts" : [agent_1_annotated_prompt_state, agent_2_annotated_prompt_state]}

def send_actions_prompts(state: PDState):
    agent_1_annotated_prompt_state = get_agent_action_annotated_prompt("agent_1", state)
    agent_2_annotated_prompt_state = get_agent_action_annotated_prompt("agent_2", state)
    return [Send("invoke_from_prompt_state_action", agent_1_annotated_prompt_state), Send("invoke_from_prompt_state_action", agent_2_annotated_prompt_state)]

def invoke_from_prompt_state_node(model, bar):
    def invoke_from_prompt_state(state : AnnotatedPrompt):
        prompt = state.prompt
        agent_name = state.agent_name
        prompt_type = state.prompt_type
        Structure = MessageResponse if prompt_type == "message" else ActionResponse
        print("invoking model...")
        response = model.with_structured_output(Structure).invoke(prompt)
        message = response.message if prompt_type == "message" else response.action #TODO this is ugly but it helps for the model to understand it s working with an action
        #print(agent_name, prompt_type, message)
        return {f"current_{prompt_type}_responses": [LLMReply(agent_name=agent_name, message=message, reply_type = prompt_type)]}
    return invoke_from_prompt_state

"""
    def map_prompts_node(prompt_type):
        def map_prompts(state: PDState):
            return [Send("invoke_from_prompt_state", prompt_state) for prompt_state in state[f"current_{prompt_type}_prompts"]] #TODO: check if this is a PromptState for sure!!!
        return map_prompts
"""

def update_state_node():
    def update_state(state: PDState) -> PDState:
        # add current message and action to actions and messages
        # from the MessageReponseState get the message and agent_name to put it as a string in the correct lists
        for message_state in state["current_message_responses"]:
            state[f"{message_state.agent_name}_messages"].append(message_state.message)
        for action_state in state["current_action_responses"]:
            state[f"{action_state.agent_name}_actions"].append(action_state.action)
        #remove current message and action
        state["current_message_responses"] = []
        state["current_action_responses"] = []
        
        # update scores
        payoff_matrix = {
            ("cooperate", "cooperate"): (3, 3),
            ("cooperate", "defect"): (0, 5),
            ("defect", "cooperate"): (5, 0),
            ("defect", "defect"): (1, 1),
        }
        agent_1_decision = state["agent_1_actions"][-1]
        agent_2_decision = state["agent_2_actions"][-1]
        score_agent1, score_agent2 = payoff_matrix[(agent_1_decision, agent_2_decision)]
        
        # add scores to scores
        state["agent_1_scores"].append(score_agent1)
        state["agent_2_scores"].append(score_agent2)
        
        #increment round
        state["current_round"] += 1    
        return state
    return update_state

def should_continue(state: PDState) -> bool:
    return (state["current_round"] <= state["total_rounds"])

def action_taken(state: PDState) -> bool:
    return (len(state["current_action_prompts"]) > 0)
    
def run_n_rounds_w_com(model_name: str, total_rounds: int, personality_key_1: str, personality_key_2: str) -> None:
    # get models
    model = get_model(model_name)
    #create graph
    graph = StateGraph(PDState, input = PDState, output = PDState)
    #add nodes
    graph.add_node("lambda_to_messages", lambda x:x)
    graph.add_node("lambda_from_messages", lambda x:x)
    graph.add_node(f"invoke_from_prompt_state_message", invoke_from_prompt_state_node(model,"ya")) #doing this because else we fall into a weird recursion state
    graph.add_node(f"invoke_from_prompt_state_action", invoke_from_prompt_state_node(model,"yo"))
    graph.add_node(f"lambda_to_actions", lambda x:x)
    graph.add_node("lambda_from_actions", lambda x:x)
    graph.add_node("update_state", update_state_node())
    
    #add edges
    graph.add_edge(START, "lambda_to_messages")
    graph.add_conditional_edges(
        source = "lambda_to_messages", 
        path = send_messages_prompts, #TODO: is this correct?? is this really a path?
        path_map = ["invoke_from_prompt_state_message"]
        )
    graph.add_edge("invoke_from_prompt_state_message","lambda_from_messages")
    #graph.add_edge("gather_messages", "generate_actions_prompts")
    graph.add_conditional_edges(
        source = "lambda_from_messages", 
        path = send_actions_prompts, #TODO: is this correct?? is this really a path?
        path_map = ["invoke_from_prompt_state_action"]
        )
    graph.add_edge("invoke_from_prompt_state_action","lambda_from_actions")
    graph.add_edge("lambda_from_actions", "update_state")
    graph.add_conditional_edges(
        source = "update_state",
        path = should_continue,
        path_map = {
            False : END,
            True : "lambda_to_messages"
            }
        )
    #compile and run
    compiled_graph = graph.compile()
    #print mermaid
    print(compiled_graph.get_graph().draw_mermaid())
    #create initial state
    game_description_prompt = get_game_description_prompt() #get from data.prompts #say how many total rounds will be played
    initial_state = PDState(
        game_description_prompt=game_description_prompt,
        personality_key_1=personality_key_1,
        personality_key_2=personality_key_2,
        current_round=1,
        total_rounds=total_rounds
    )
    end_state = compiled_graph.invoke(initial_state)
    return end_state