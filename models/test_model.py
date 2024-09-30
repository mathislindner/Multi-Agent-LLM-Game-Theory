from langchain_core.language_models.llms import LLM

#model that replies with "test" to any input
class TestModel(LLM):
    def __init__(self):
        super().__init__()
        self.name = "TestModel"
        self.description = "A model that replies with 'test' to any input"
        self.input_schema = {
            "input": {
                "type": "string",
                "description": "The input to the model"
            }
        }
        self.output_schema = {
            "output": {
                "type": "string",
                "description": "The output of the model"
            }
        }
    
    def _run(self, input: str) -> str:
        return "test"