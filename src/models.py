import logging
from dotenv import load_dotenv
import os

#TODO: use init chat model (can also infer provider from jsut model name)
from langchain.chat_models import init_chat_model


#https://python.langchain.com/docs/integrations/chat/

#set all env vars to have the same name automatically
load_dotenv()
for k, v in os.environ.items():
    os.environ[k] = v
#--------------------------------------------------------

#-------------------------------------------------------
def get_model_by_id_and_provider(model_id: str, provider: str = None):
    properties = {
        "temperature": 0,
        "max_retries": 3
        #"timeout": 2000
    }
    if provider != "anthropic":
        properties["seed"] = 42  # seed doesn’t work with anthropic
        
    if provider is None:
        return init_chat_model(
            model_id,
            **properties
        )
    return init_chat_model(
        model_id,
        model_provider=provider,
        **properties
    )