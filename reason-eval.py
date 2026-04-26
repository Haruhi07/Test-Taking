from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import os
import json
from ast import literal_eval
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from utils import dig_answers

def process_file(p):
    df = pd.read_csv(p)
    df = df[df.label != df.preds]
    df.to_csv("WA.csv", index=False)


def main(args):
    if os.path.isfile(args.folder_path):
        process_file(args.folder_path)
        return
    avg = 0
    avg_num = 0
    for file_name in os.listdir(args.folder_path):
        file_path = os.path.join(args.folder_path, file_name)
        df = pd.read_csv(file_path)
        if "factax" in file_path and file_name != "diasumfact.csv":
            df = df[df.cut == "test"]
        labels = df["label"]
        if "preds" not in df.columns.to_list():
            preds = []
            for reasons_str in df["reasons"].tolist():
                reasons = literal_eval(reasons_str)
                preds.append(dig_answers(reasons))
            df["preds"] = preds
            df.to_csv(file_path, index=False)
        else:
            preds = df["preds"]
        score = balanced_accuracy_score(labels, preds)
        confusion_mat = confusion_matrix(labels, preds)
        print(f"{file_name}: {100 * score:.2f} -- len: {len(df)}")
        #print(confusion_mat)
        if ("train" in file_name) or ("test" in file_name):
            continue
        avg += score
        avg_num += 1
    if avg_num > 0:
        print(f"Average: {100 * avg / avg_num:.2f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    args = parser.parse_args()

    main(args)