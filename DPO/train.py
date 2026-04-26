from vllm import LLM, SamplingParams
from argparse import ArgumentParser
import logging
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm
import os
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer
import shutil


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(args):

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=True)
    def count_len(data):
        input_ids = tokenizer(data["prompt"])["input_ids"]
        data["prompt_length"] = len(input_ids)
        return data
    dataset_path = args.dataset_path
    logging.info(f"Loading dataset: {dataset_path}")
    dataset = Dataset.from_csv(f"{args.dataset_path}")
    train_dataset, test_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()

    train_dataset = train_dataset.map(count_len, num_proc=4)
    test_dataset = test_dataset.map(count_len, num_proc=4)

    train_dataset = train_dataset.filter(lambda x: x["prompt_length"] < 8000).shuffle()
    test_dataset = test_dataset.filter(lambda x: x["prompt_length"] < 8000)

    logging.info(f"Train dataset length: {len(train_dataset)}")
    logging.info(f"Test dataset length: {len(test_dataset)}")

    logging.info(f"Output will be saved to: {args.output_path}")
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)

    training_args = DPOConfig(
        output_dir = args.output_path,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=64,
        num_train_epochs=3,
        max_length=10000,
        max_prompt_length=8000,
        dataset_num_proc=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-7,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        beta=0.01,
        bf16=True,
        optim="adamw_torch_fused",
        weight_decay=0.05,
        max_grad_norm=1.0,
        eval_steps=100,
        eval_strategy="steps",
        save_steps=100,
        save_strategy="steps",
        save_total_limit=3,
        logging_dir=f"{args.output_path}/log",
        logging_steps=10,
        report_to="tensorboard",
        eval_on_start=True,
        max_steps=300,
    )



    trainer = DPOTrainer(
        model=args.model_name,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for processing.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset for processing.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed dataset.")
    args = parser.parse_args()
    main(args)