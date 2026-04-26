from argparse import ArgumentParser
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import shutil
import logging
import os
import sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils import CLAIM_PROMPT

def preprocess(data):
    prompt = CLAIM_PROMPT.replace("<claim>", data["claim"])
    completion = data["facts"]
    return {"prompt": prompt, "completion": completion}

def main(args):
    logging.info(f"Loading dataset: {args.dataset_path}")
    raw_train_set = Dataset.from_csv(os.path.join(args.dataset_path, "train.csv"))
    raw_test_set = Dataset.from_csv(os.path.join(args.dataset_path, "test.csv"))

    train_set = raw_train_set.map(preprocess)
    test_set = raw_test_set.map(preprocess)

    model_name = "Qwen/Qwen3-0.6B"
    logging.info(f"Training model: {model_name}")
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)

    trainin_args = SFTConfig(
        completion_only_loss=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=10,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        chat_template_path=model_name,
        max_length=None,
        eval_strategy="epoch",
        logging_dir=f"{args.output_path}/logs",
        logging_steps=10,
        output_dir=args.output_path,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="tensorboard",
    )

    sft_trainer = SFTTrainer(
        model=model_name,
        args=trainin_args,
        train_dataset=train_set,
        eval_dataset=test_set,
    )
    sft_trainer.train()
    sft_trainer.save_model(f"{args.output_path}/best_model")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(args)