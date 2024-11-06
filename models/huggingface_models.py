import torch
from transformers import pipeline
import os
import dotenv

dotenv.load_dotenv()

def get_huggingface_pipeline(MODEL_ID):
    HUGGINFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    os.environ["HF_HOME"] = "/cluster/scratch/mlindner/cache/"
    # if gpu available use it
    device = 0 if torch.cuda.is_available() else -1
    print("loading model on device: ", device)
    if os.path.exists("/cluster/scratch/mlindner/master_thesis/data/models/{}/".format(MODEL_ID)):
        pipe = pipeline("text-generation", model="/cluster/scratch/mlindner/master_thesis/data/models/{}/".format(MODEL_ID), torch_dtype=torch.bfloat16, device=device, token=HUGGINFACE_TOKEN)
    else:
        pipe = pipeline(
            "text-generation", 
            model=MODEL_ID, 
            torch_dtype=torch.bfloat16,
            device=device,
            token=HUGGINFACE_TOKEN,
            use_cache=False,
            #load_in_4bit=True
        )
        pipe.save_pretrained("/cluster/scratch/mlindner/master_thesis/data/models/{}/".format(MODEL_ID))
    return pipe

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