from argparse import ArgumentParser
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import shutil
import logging
import os
import sys
from transformers import AutoTokenizer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def main(args):
    logging.info(f"Training model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logging.info(f"Loading dataset: {args.dataset_path}")
    def preprocess(data):
        prompt_ids = tokenizer(data["prompt"])["input_ids"]
        data["prompt_len"] = len(prompt_ids)
        return data
    
    train_set = Dataset.from_csv(f"{args.dataset_path}/train.csv")
    test_set = Dataset.from_csv(f"{args.dataset_path}/test.csv")

    test_set = test_set.map(preprocess, num_proc=8)
    test_set = test_set.filter(lambda x: x["prompt_len"] <= 2000).shuffle(seed=42)
    train_set = train_set.map(preprocess, num_proc=8)
    train_set = train_set.filter(lambda x: x["prompt_len"] <= 2000)


    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)

    trainin_args = SFTConfig(
        completion_only_loss=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        num_train_epochs=5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        chat_template_path=args.model_name,
        eval_strategy="epoch",
        logging_dir=f"{args.output_path}/logs",
        logging_steps=10,
        output_dir=args.output_path,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="tensorboard",
        eval_on_start=True,
    )

    sft_trainer = SFTTrainer(
        model=args.model_name,
        args=trainin_args,
        train_dataset=train_set,
        eval_dataset=test_set,
    )
    sft_trainer.train()
    sft_trainer.save_model(f"{args.output_path}/best_model")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(args)