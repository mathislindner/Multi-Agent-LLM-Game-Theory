from langchain.llms.base import LLM
from typing import Optional
import random
# Define a mock LLM class that always returns "test"
class TestModel(LLM):
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