import os
import pandas as pd
import evaluate
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer, util


class SECS:
    def __init__(self):
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def compute(self, predictions, references):
        pred_embeddings = self.encoder.encode(predictions, convert_to_tensor=True)
        ref_embeddings = self.encoder.encode(references, convert_to_tensor=True)
        cos_scores = util.cos_sim(pred_embeddings, ref_embeddings)
        return cos_scores.max(dim=1)[0].mean().item()


def secs(scorer, preds, refs):
    ave = 0
    for pred, ref in zip(preds, refs):
        pred_list = [fact.strip("- ") for fact in pred.split("\n")]
        ref_list = [fact.strip("- ") for fact in ref.split("\n")]
        score = scorer.compute(predictions=pred_list, references=ref_list)
        ave += score
    ave /= len(preds)
    return ave



def main(args):
    rouge = evaluate.load("rouge")
    secs_scorer = SECS()
    for f in os.listdir(args.reference_path):
        ref_file = os.path.join(args.reference_path, f)
        pred_file = os.path.join(args.prediction_path, f)
        reference_df = pd.read_csv(ref_file)
        prediction_df = pd.read_csv(pred_file)
        references = reference_df["facts"]
        predictions = prediction_df["facts"]
        scores = rouge.compute(references=references, predictions=predictions)
        print(f"{f}: {scores}")
        secs_score = secs(secs_scorer, predictions, references)
        print(f"{f} SECS: {secs_score}")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--prediction_path", type=str, required=True)
    args = parser.parse_args()

    main(args)