from argparse import ArgumentParser
import logging
import pandas as pd
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
import os


def main():
    factax_train = Dataset.from_csv("/home/b5ag/yuxuan.b5ag/project/0.6B/decompose-train/generated_facts/factax/train.csv").remove_columns(["origin", "id", "dataset"])
    factax_test = Dataset.from_csv("/home/b5ag/yuxuan.b5ag/project/0.6B/decompose-train/generated_facts/factax/test.csv").remove_columns(["origin", "id", "dataset"])

    c2d_train = Dataset.from_csv("/home/b5ag/yuxuan.b5ag/project/0.6B/decompose-train/generated_facts/c2d/train.csv")
    c2d_test = Dataset.from_csv("/home/b5ag/yuxuan.b5ag/project/0.6B/decompose-train/generated_facts/c2d/test.csv")

    d2c_train = Dataset.from_csv("/home/b5ag/yuxuan.b5ag/project/0.6B/decompose-train/generated_facts/d2c/train.csv")
    d2c_test = Dataset.from_csv("/home/b5ag/yuxuan.b5ag/project/0.6B/decompose-train/generated_facts/d2c/test.csv")

    train_set = concatenate_datasets([factax_train, c2d_train, d2c_train])
    test_set = concatenate_datasets([factax_test, c2d_test, d2c_test])

    train_set.to_csv(f"/home/b5ag/yuxuan.b5ag/project/0.6B/decompose-train/generated_facts/dataset/train.csv", index=False)
    test_set.to_csv(f"/home/b5ag/yuxuan.b5ag/project/0.6B/decompose-train/generated_facts/dataset/test.csv", index=False)

    print(f"{len(train_set)} pieces of data in the training set.")
    print(f"{len(test_set)} pieces of data in the test set.")



if __name__ == "__main__":
    main()