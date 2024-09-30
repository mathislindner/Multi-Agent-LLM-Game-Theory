from langchain_core.tools import BaseTool
import json
from langchain_core.tools import tool
from pydantic import BaseModel

#This is the tool that gives the option for the player to make a decision in the Prisoner's Dilemma game to cooperate or defect
@tool
class PrisonersDilemmaDecision(BaseTool):
    name:str = "PrisonersDilemmaDecision"
    description:str = "Takes the decision of a player in the Prisoner's Dilemma game: to cooperate or defect"

    def _run(self, decision: str):
        #checks if the decision is valid
        if decision not in ["cooperate", "defect"]:
            raise ValueError("The decision must be either 'cooperate' or 'defect'")
        # add the decision to the 
        return decision

#this is technically not really a tool: it is not meant for the agents to use, but for the game engine to use. I just put it here to use it in langgraph
@tool
class PrisonersDilemmaPayoff(BaseTool):
    name:str = "Calculate Payoffs from both players"
    description:str = "Calculate the payoffs for both players in the Prisoner's Dilemma game"

    def _run(self, decisions) -> dict:
        #loads the payoff matrix from the json file
        payoff_matrix = json.load(open("logic/payoff_matrix_prison_dillema.json"))
        #get the payoffs for the decisions
        payoffs = payoff_matrix[tuple(decisions)]
        return payoffs
    
