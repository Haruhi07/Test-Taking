import pandas as pd
import sys
import os

def combine_csv_files(file1, file2, output_file):
    """
    Combines two CSV files into a single output CSV file.
    """
    try:
        if not os.path.exists(file1):
            raise FileNotFoundError(f"File not found: {file1}")
        if not os.path.exists(file2):
            raise FileNotFoundError(f"File not found: {file2}")
            
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        combined_df = pd.concat([df1, df2], ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully combined files into {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
        split = "test"
        file1 = f"/home/b5ag/yuxuan.b5ag/project/0.6B/reason-train/factax-dataset/{split}.csv"
        file2 = f"/home/b5ag/yuxuan.b5ag/project/0.6B/llm_data/normal/{split}.csv"
        output_file = f"/home/b5ag/yuxuan.b5ag/project/0.6B/reason-train/dataset/{split}.csv"
        combine_csv_files(file1, file2, output_file)