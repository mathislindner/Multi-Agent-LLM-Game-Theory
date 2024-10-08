import torch
from transformers import pipeline
import os
import dotenv

dotenv.load_dotenv()

HUGGINFACE_TOKEN = os.getenv("HUGGINFACE_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")
#right now model not saved locally

def get_huggingface_pipeline(model_id):
    return pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        token=HUGGINFACE_TOKEN,
    )

def try_huggingface_model():
    pipe = get_huggingface_pipeline(MODEL_ID)
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=5,
    )
    print(outputs[0]["generated_text"][-1])