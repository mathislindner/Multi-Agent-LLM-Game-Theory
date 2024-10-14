import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import dotenv
import os
import argparse
dotenv.load_dotenv()

def launch_test(model_id):
    HUGGINFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    MODEL_ID = model_id
    # if gpu available use it
    device = 0 if torch.cuda.is_available() else -1
    print("device: ", device)

    if os.path.exists("/cluster/scratch/mlindner/master_thesis/data/models/{}/".format(MODEL_ID)):
        #pipe = pipeline("text-generation", model="/cluster/scratch/mlindner/master_thesis/data/models/{}/".format(MODEL_ID), torch_dtype=torch.bfloat16, device=device, token=HUGGINFACE_TOKEN)
        tokenizer = AutoTokenizer.from_pretrained("/cluster/scratch/mlindner/master_thesis/data/models/{}/".format(MODEL_ID))
        model = AutoModelForCausalLM.from_pretrained("/cluster/scratch/mlindner/master_thesis/data/models/{}/".format(MODEL_ID))
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        """pipe = pipeline(
            model_type="text-generation",
            model=MODEL_ID,
            device=device,
            token=HUGGINFACE_TOKEN,
            cache_dir="/cluster/scratch/mlindner/cache/",
        )"""

    #pipe.save_pretrained("/cluster/scratch/mlindner/master_thesis/data/models/{}/".format(MODEL_ID))

    #generate example
    prompt = "You are a pirate chatbot who always responds in pirate speak!"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, do_sample=True)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="The model id to use for the game", required=True)
    args = parser.parse_args()
    launch_test(model_id=args.model_id)