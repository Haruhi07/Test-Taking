from vllm import LLM, SamplingParams
from argparse import ArgumentParser
import logging
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm
import os
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CLAIM_PROMPT ="""Follwing the example below, segment the given claim into atomic facts only based on the claim itself. Output each fact with \"-\" as the start.

Claim:
The parkway was opened in 2001 after just under a year of construction and almost two decades of community requests.
Facts:
- The parkway was opened in 2001.
- The parkway was opened after just under a year of construction.
- The parkway was opened after two decades of community requests.

CLAIM:
<claim>
Facts:
"""

DOC_PROMPT = """Read the article given below and answer the questions.

ARTICLE:
<article>

Here is a claim, answer the following questions. Please reason step by step, and output your final answer by "Final Answer: yes" or "Final Answer: no".

CLAIM:
<claim>

1) For the claim, are the object and the subject mentioned?

2) If the object and the subject are mentioned, is their related information verifiable according to the article? If there is information not mentioned, carry it into the next question. If verifiable but incorrect, stop here and answer "Final Answer: no".

3) Look at the relationships between the object and the subject, is their relationship mentioned? If not, can the relationship be inferred from the article? If the relationship stands, can the previous information not mentioned be inferred from the article?
"""

ChatGPT_PROMPT = """Read the article given below and answer the questions.

ARTICLE:
<article>

Here is a claim, answer the following questions. Please reason step by step, and output your final answer by "Final Answer: yes" or "Final Answer: no".

CLAIM:
<claim>

1) Does the claim explicitly mention both the subject and the object?

2) If both are mentioned, can the information connecting them be verified using the article? If some details are missing, carry those forward to the next step. If the information is verifiable but incorrect, stop and respond with “Final Answer: no.”

3) Examine the relationship between the subject and the object. Is this relationship stated in the article? If not, can it reasonably be inferred? If the relationship is valid, determine whether any previously missing information can also be inferred from the article.
"""

Gemini_PROMPT = """Read the article given below and answer the questions.

ARTICLE:
<article>

Here is a claim, answer the following questions. Please reason step by step, and output your final answer by "Final Answer: yes" or "Final Answer: no".

CLAIM:
<claim>

1) Does the claim explicitly identify both the subject and the object?

2) If both entities are present, is the provided detail supported by the text? If the data is present but contradicts the source, mark as "no" and terminate. If details are missing but not contradicted, proceed to the next step.

3) Is the connection between the entities stated or clearly implied? If a relationship exists, determine if any previously missing details can be logically deduced from the source material.
"""

NO_Inference_PROMPT = """
Read the article given below and answer the questions.

ARTICLE:
<article>

Here is a claim, answer the following questions. Please reason step by step, and output your final answer by "Final Answer: yes" or "Final Answer: no".

CLAIM:
<claim>

1) For the claim, are the object and the subject mentioned?

2) If the object and the subject are mentioned, is their related information verifiable according to the article? If there is information not mentioned, carry it into the next question. If verifiable but incorrect, stop here and answer "Final Answer: no".

3) Look at the relationships between the object and the subject, is their relationship explicitly mentioned? If explicitly mentioned, answer "Final Answer: yes", otherwise answer "Final Answer: no".
"""

Only_Inference_PROMPT = """
Read the article given below and answer the questions.

ARTICLE:
<article>

Here is a claim, answer the following questions. Please reason step by step, and output your final answer by "Final Answer: yes" or "Final Answer: no".

CLAIM:
<claim>

Is the claim supported by the article? You can do reasonable inference if needed. Answer "Final Answer: yes" or "Final Answer: no".
"""

SINGLE_PASS_PROMPT = """
Follwing the example below, segment the given claim into atomic facts only based on the claim itself first. Output each fact with \"-\" as the start.

Claim:
The parkway was opened in 2001 after just under a year of construction and almost two decades of community requests.
Facts:
- The parkway was opened in 2001.
- The parkway was opened after just under a year of construction.
- The parkway was opened after two decades of community requests.

CLAIM:
<claim>

Read the article given below. For each fact, answer the questions given below. Please reason step by step, and output your final answer by "Final Answer: yes" or "Final Answer: no".

ARTICLE:
<article>

Questions:

1) For the claim, are the object and the subject mentioned?

2) If the object and the subject are mentioned, is their related information verifiable according to the article? If there is information not mentioned, carry it into the next question. If verifiable but incorrect, stop here and answer "Final Answer: no".

3) Look at the relationships between the object and the subject, is their relationship mentioned? If not, can the relationship be inferred from the article? If the relationship stands, can the previous information not mentioned be inferred from the article?
"""

def batch_data(st, dataset, batch_size=64):
    data_length = len(dataset)
    for i in range(st, data_length, batch_size):
        yield dataset[i:i + batch_size]

def summarise_answers(answer_list):
    for answer in answer_list:
        if "no" in answer:
            return 0
    return 1


def main(args):
    dataset_path = args.dataset_path
    logging.info(f"Loading dataset: {dataset_path}")
    if dataset_path == "lytang/LLM-AggreFact":
        dataset = load_dataset("lytang/LLM-AggreFact", split="test")
        dataset = dataset.filter(lambda x: x["dataset"] == args.subset)

        save_filename = f"{args.subset}.csv"
    elif "aggre_fact_sota.csv" in dataset_path:
        dataset = Dataset.from_csv(dataset_path)
        dataset = dataset.rename_columns({"summary": "claim"})
        dataset = dataset.remove_columns(["DAE_score", "DAE_label", "QuestEval_score", "QuestEval_label", "SummaC-ZS_score", "SummaC-ZS_label", "SummaC-Conv_score","SummaC-Conv_label","QAFactEval_score", "QAFactEval_label"])
        dataset = dataset.filter(lambda x: x["dataset"] == args.subset)
        if args.origin is not None:
            dataset = dataset.filter(lambda x: x["origin"] == args.origin)

        save_filename = f"{args.subset}.csv" if args.origin is None else f"{args.subset}-{args.origin}.csv"
    elif "diasumfact.csv" in dataset_path:
        dataset = Dataset.from_csv(dataset_path)
        dataset = dataset.rename_columns({"error": "label"})
        dataset = dataset.remove_columns(["entity", "predicate", "circumstance", "coreference", "additional", "others", "error_type" ,"dataset"])

        save_filename = "diasumfact.csv"
    elif "WA_" in dataset_path:
        dataset = Dataset.from_csv(dataset_path)
        save_filename = f"new_WA_{args.subset}.csv"
    elif "new_factax-test.csv" in dataset_path:
        dataset = Dataset.from_csv(dataset_path)
        save_filename = "new_factax-test.csv"

    
    save_path = os.path.join(args.output_path, save_filename)

    def save_preds(saved_preds, results):
        preds = [summarise_answers(answers) for answers in results]
        df = pd.DataFrame(dataset).iloc[:len(saved_preds + preds)]
        df["preds"] = saved_preds + preds
        
        df.to_csv(save_path, index=False)

    model_name = args.model_path
    logging.info(f"Loading model: {model_name}")
    #llm = LLM(model=model_name, max_model_len=40000, disable_cascade_attn=True, seed=17)
    llm = LLM(model=model_name, max_model_len=40000, disable_cascade_attn=True, seed=71)
    #llm = LLM(model=model_name, max_model_len=40000, disable_cascade_attn=True, seed=13)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    claim_params = SamplingParams(max_tokens=500)
    reason_params = SamplingParams(max_tokens=30000)

    claim_messages = []
    claim_input_prompts = []
    for i in range(len(dataset)):
        claim = dataset[i]["claim"]
        claim_task = CLAIM_PROMPT.replace("<claim>", claim)
        claim_messages.append([{"role": "user", "content": claim_task}])
    for message in claim_messages:
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        claim_input_prompts.append(prompt)
    
    decompose_outputs = llm.generate(claim_input_prompts, claim_params, use_tqdm=False)
    generated_facts = [output.outputs[0].text for output in decompose_outputs]
    #generated_facts = [dataset[i]["claim"] for i in range(len(dataset))]

    assert len(generated_facts) == len(dataset)

    judge_messages = []
    judge_input_prompts = []
    for i in range(len(dataset)):
        facts = generated_facts[i]
        doc = dataset[i]["doc"]
        fact_list = [fact.strip("- ") for fact in facts.split("\n")]
        for fact in fact_list:
            #doc_tasks = NO_Inference_PROMPT.replace("<article>", doc).replace("<claim>", fact)
            doc_tasks = Only_Inference_PROMPT.replace("<article>", doc).replace("<claim>", fact)
            judge_messages.append([{"role": "user", "content": doc_tasks}])
    for message in judge_messages:
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        judge_input_prompts.append(prompt)
    
    judge_outputs = llm.generate(judge_input_prompts, reason_params, use_tqdm=True)
    judge_answers = [output.outputs[0].text for output in judge_outputs]
    judge_results = [answer.split("Final Answer: ")[-1].strip(" .*") for answer in judge_answers]

    st = 0
    results = []

    for i in range(len(dataset)):
        facts = generated_facts[i]
        fact_list = [fact.strip("- ") for fact in facts.split("\n")]
        fact_num = len(fact_list)
        batch_results = judge_results[st: st+ fact_num]
        results.append(summarise_answers(batch_results))
        st += fact_num
        
    df = pd.DataFrame(dataset)
    df["preds"] = results
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model for processing.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset for processing.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed dataset.")
    parser.add_argument("--subset", type=str, default=None, help="Which subset to process.")
    parser.add_argument("--origin", type=str, default=None, help="Which origin to process.")
    args = parser.parse_args()
    main(args)