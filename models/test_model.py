from typing import Optional
import random

class TestModel:
    def __init__(self):
        pass

    def invoke(self, prompt: str) -> str:
        return random.choice(["I will cooperate", "I will defect"])

"""from langchain.llms.base import LLM
# Define a mock LLM class that always returns "test"
class LangchainTestModel(LLM):
    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        return random.choice(["cooperate", "defect"])
    
    @property
    def _llm_type(self) -> str:
        return "mock-llm"

    @property
    def _identifying_params(self) -> dict:
        return {"model": "mock-llm"}

def test_mock_llm():
    # Instantiate the mock LLM
    llm = TestModel()

    # Test the LLM by calling it with any prompt using `invoke`
    response = llm.invoke("Any input")
    print(response)  # Output will be either "cooperate" or "defect"
    """