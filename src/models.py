import logging
from dotenv import load_dotenv
import os

#TODO: use init chat model (can also infer provider from jsut model name)
from langchain.chat_models import init_chat_model

#https://python.langchain.com/docs/integrations/chat/
#all these are possible models
load_dotenv()
#--------------------------------------------------------
def get_model(model_name: str):
    #if name starts with gpt..
    if model_name.startswith("gpt"):
        from langchain_openai import ChatOpenAI
        logging.info(f"Creating OpenAI model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            seed=42,
            max_retries=2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    if model_name.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        logging.info(f"Creating Anthropic model: {model_name}")
        return ChatAnthropic(
            model=model_name,
            temperature=0,
            seed=42,
            max_retries=2,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    if model_name.startswith("mistral"):
        from langchain_mistral import ChatMistral
        logging.info(f"Creating Mistral model: {model_name}")
        return ChatMistral(
            model=model_name,
            temperature=0,
            seed=42,
            max_retries=2,
            api_key=os.getenv("MISTRAL_API_KEY")
        )
    if model_name.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        logging.info(f"Creating Google model: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            seed=42,
            max_retries=2,
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
    
    else: #assuming it s a hugginface model
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from torch.cuda import is_available
        HF_HOME = os.getenv("HF_HOME") #set cache directory
        HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
        device = 0 if is_available() else -1
        #tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                device=device,        
                pipeline_kwargs=dict(
                    repetition_penalty=1.0,
                ),
            )            
            llm.pipeline.tokenizer.pad_token = llm.pipeline.tokenizer.eos_token
            hf_chat = ChatHuggingFace(llm = llm)
        except Exception as e:
            raise(f"Error in creating HuggingFace model: {e}")
        return hf_chat
    