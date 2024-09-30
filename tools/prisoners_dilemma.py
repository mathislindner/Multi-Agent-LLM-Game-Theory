from langchain_core.tools import BaseTool
import json
from langchain_core.tools import tool
from pydantic import BaseModel

@tool
class PrisonersDilemmaDecision(BaseTool):
    name:str = "Takes the decision of a player in the Prisoner's Dilemma game"
    description:str = "Takes the decision of a player in the Prisoner's Dilemma game"
    args_schema:json = {
        "decision": {
            "type": "boolean",
            "description": "The decision of the player in the Prisoner's Dilemma game"
        }
    }
    def _run(self, decision: bool):
            return decision
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
    
