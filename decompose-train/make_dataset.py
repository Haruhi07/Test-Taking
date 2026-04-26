import os
import sys
from datasets import Dataset, load_dataset
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils import CLAIM_PROMPT


def batch_data(st, dataset, batch_size=32):
    data_length = len(dataset)
    for i in range(st, data_length, batch_size):
        yield dataset[i:i + batch_size]

def generate_facts(dataset, split, output_path):
    batch_generated_facts = []

    for batch in tqdm(batch_data(0, dataset, 32), desc="Processing data", total=len(dataset) // 32 + 1):
        claim_tasks = [CLAIM_PROMPT.replace("<claim>", claim) for claim in batch["claim"]]
        messages = [[{"role": "user", "content": claim_prompt}] for claim_prompt in claim_tasks]
        input_prompts = []
        for message in messages:
            prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            input_prompts.append(prompt)
        decompose_outputs = llm.generate(input_prompts, claim_params, use_tqdm=False)
        generated_facts = [output.outputs[0].text for output in decompose_outputs]
        batch_generated_facts.extend(generated_facts)

    df = pd.DataFrame(dataset)
    df["facts"] = batch_generated_facts
    df.to_csv(f"{output_path}/{split}.csv", index=False)

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    llm = LLM(model=model_name, max_model_len=30000, disable_cascade_attn=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    claim_params = SamplingParams(max_tokens=500)

    #dataset_path = "/home/b5ag/yuxuan.b5ag/project/data/dataset.csv"
    dataset_path = "lytang/C2D-and-D2C-MiniCheck"
    split = "d2c" #"c2d" or "d2c"
    if dataset_path == "lytang/C2D-and-D2C-MiniCheck":
        dataset = load_dataset(dataset_path, split=split)
        save_path = f"generated_facts/{split}"
    else:
        columns_to_remove = ["model_name", "cut", "error_type", "annotated_span", "entity", "predicate", "additional", "circumstance", "coreference", "others"]
        full_dataset = Dataset.from_csv(dataset_path)
        existing_columns_to_remove = [col for col in columns_to_remove if col in full_dataset.column_names]
        dataset = full_dataset.remove_columns(existing_columns_to_remove)
        dataset = dataset.rename_columns({"error": "label"})
        save_path = "generated_facts/factax"
    
    dataset = dataset.train_test_split(train_size=0.9, shuffle=True, seed=42)

    generate_facts(dataset["train"], "train", save_path)
    generate_facts(dataset["test"], "test", save_path)
