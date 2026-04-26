import pandas as pd
from ast import literal_eval
import os, sys
from datasets import Dataset
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from utils import DOC_PROMPT

if __name__ == "__main__":
    df = pd.read_csv("/home/b5ag/yuxuan.b5ag/project/0.6B/reason-train/annotations/train-small.csv")

    df = df[df.preds == df.label]
    df = df[df.preds != df.anno_preds]

    print(f"{len(df)} pieces of data left")

    docs = df.doc.tolist()
    facts = df.facts.tolist()
    reasons = df.reasons.tolist()
    annos = df.annos.tolist()
    prompt_list, chosen_list, reject_list = [], [], []
    for doc, fact_str, A3B_reason_str, small_reason_str in zip(docs, facts, reasons, annos):
        fact_list = [fact.strip("- ") for fact in fact_str.split("\n")]
        doc_task = DOC_PROMPT.replace("<article>", doc)
        prompts = [doc_task.replace("<claim>", fact) for fact in fact_list]
        A3b_reasons = literal_eval(A3B_reason_str)
        small_reasons = literal_eval(small_reason_str)
        for prompt, A3B_reason, small_reason in zip(prompts, A3b_reasons, small_reasons):
            prompt_list.append(prompt)
            chosen_list.append(A3B_reason)
            reject_list.append(small_reason)
    
    dataset = Dataset.from_dict({"prompt": prompt_list, "chosen": chosen_list, "rejected": reject_list})
    train_set, test_set = dataset.train_test_split(test_size=0.1, seed=42).values()
    train_df = train_set.to_pandas()
    test_df = test_set.to_pandas()

    train_df.to_csv("/home/b5ag/yuxuan.b5ag/project/0.6B/reason-train/dataset/train-dpo.csv", index=False)