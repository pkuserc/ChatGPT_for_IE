import json
import time
import openai
import os
from tqdm import tqdm
import argparse 

openai.api_key = ""
result_dir = './Output'


def call_openai(args):
    saved_dir = os.path.join(result_dir, args.task_dataset)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    with open(f"./Prompt/{args.task_dataset}.json", "r", encoding="utf-8") as f:
        lines = json.load(f)

    bar = tqdm(lines.items())
    for idx, prompt in bar:
        bar.set_description("Running")
        processed_idx = [f[:-5] for f in os.listdir(saved_dir)]
        if idx not in processed_idx:

            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )["choices"][0]["message"]["content"]
                    break
                except:
                    bar.set_description("Sleeping")
                    time.sleep(3)

            with open(os.path.join(saved_dir, f"{idx}.json"), "w") as writer:
                writer.write(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_dataset", 
        type=str, 
        required=True,
        choices=['EAE_E_Closed','EAE_E_Open','EAE_E+_Closed','EAE_E+_Open','ED_E_Closed','ED_E_Open','ED_E+_Closed','ED_E+_Open','EE_E_Closed','EE_E_Open','EE_E+_Closed','EE_E+_Open']
    )
    args = parser.parse_args()

    call_openai(args)