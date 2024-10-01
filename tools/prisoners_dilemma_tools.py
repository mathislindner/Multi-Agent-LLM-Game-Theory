from langchain_core.tools import tool

@tool
def prisoners_dilemma_communication(agent_name: str, message: str) -> str:
    """
    Allows agents to communicate their strategy in the game.
    
    Args:
        agent_name (str): The name of the agent communicating.
        message (str): The communication message from the agent.
        
    Returns:
        str: A formatted string with the agent's communication.
    """
    return f"{agent_name} communicated: {message}"

@tool
def prisoners_dilemma_decision(agent_name: str, decision: str) -> str:
    """
    Allows agents to make a decision in the game.
    
    Args:
        agent_name (str): The name of the agent making the decision.
        decision (str): The decision made by the agent (either 'cooperate' or 'defect').
        
    Returns:
        str: A formatted string with the agent's decision.
    """
    return f"{agent_name} decided to {decision}"
