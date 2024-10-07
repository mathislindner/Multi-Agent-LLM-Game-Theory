import torch
from transformers import pipeline
import os
import dotenv

dotenv.load_dotenv()
HUGGINFACE_TOKEN = os.getenv("HUGGINFACE_TOKEN")

#right now model not saved locally
model_id = "meta-llama/Llama-3.2-1B-Instruct"
model_id = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HUGGINFACE_TOKEN,
)

pipe.save_pretrained("data/models/llama1b-4bit")

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=5,
)
print(outputs[0]["generated_text"][-1])