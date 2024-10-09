import torch
from transformers import pipeline
import dotenv
import os
dotenv.load_dotenv()

HUGGINFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")
# if gpu available use it
device = 0 if torch.cuda.is_available() else -1
print("device: ", device)

if os.path.exists("/cluster/scratch/mlindner/master_thesis/data/models/{}/".format(MODEL_ID)):
    pipe = pipeline("text-generation", model="/cluster/scratch/mlindner/master_thesis/data/models/{}/".format(MODEL_ID), torch_dtype=torch.bfloat16, device=device, token=HUGGINFACE_TOKEN)

else:
    pipe = pipeline(
        "text-generation", 
        model=MODEL_ID, 
        torch_dtype=torch.bfloat16,
        device=device,
        token=HUGGINFACE_TOKEN,
    )

pipe.save_pretrained("/cluster/scratch/mlindner/master_thesis/data/models/{}/".format(MODEL_ID))

print(pipe("The key to life is"))