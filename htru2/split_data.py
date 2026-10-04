import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    """ 
    1. parser.add_argument(...)
    This method tells the argparse library to listen for a specific flag typed in the terminal.

    2. '--data' and '--out' (The Flag Names)
    These strings define the exact names of the flags the user must type.

    --data is meant to accept the path to your source dataset (e.g., HTRU_2.csv).
    --out is meant to accept the name of the folder where you want to save the split files (e.g., split_dataset).
    3. required=True (Making them Mandatory)
    By setting required=True, you are telling the script that it must crash and throw an error if the user forgets to provide these flags. It prevents the script from running blindly without knowing where to find the data or where to save it.

    How it connects to the command you ran earlier:
    When you ran this command in the terminal:

-----------------------------------
    bash
    python split_data.py --data HTRU_2.csv --out split_dataset

------------------------------------

    Here is exactly what happened behind the scenes:
    The script saw --data HTRU_2.csv. Because of parser.add_argument('--data'), it stored the string "HTRU_2.csv" into a variable named args.data.
    The script saw --out split_dataset. Because of parser.add_argument('--out'), it stored the string "split_dataset" into a variable named args.out.
    """

    # Load data
    df = pd.read_csv(args.data, header=None)
    
    # Split: 70% train, 15% val, 15% test
    # First, split into train (70%) and temp (30%)
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df.iloc[:, -1], random_state=42)
    
    # Then, split temp into val (50% of 30% = 15%) and test (15%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df.iloc[:, -1], random_state=42)

    """ 
    stratify=df.iloc[:, -1] (Balancing the Classes)
    This is arguably the most crucial argument for your specific dataset:

    df.iloc[:, -1] grabs the very last column of your dataset (which contains the 0 or 1 indicating if it's a pulsar).
    By passing this to stratify, we are telling the function: "Look at these labels. Make sure the percentage of pulsars (9.16%) is exactly the same in both the 70% chunk and the 30% chunk."
    Without stratify, the random split might accidentally put 15% pulsars in the train set and only 2% in the test set, which would ruin your model's ability to learn and evaluate properly.
    
    """

    """ 
    random_state=42 (The Seed for Reproducibility)
    Under the hood, train_test_split shuffles your rows randomly before cutting them.

    Computers generate "random" numbers using mathematical formulas that start from a "seed" number.
    By explicitly setting the seed to 42 (or any fixed integer), we guarantee that the random shuffling happens exactly the same way every single time the script runs.
    If you were to remove this argument, you would get slightly different data in your train, val, and test sets every time you ran the script.
     
    """

    # Ensure the directory exists
    os.makedirs(args.out, exist_ok=True)
    
    # Save the files
    train_df.to_csv(os.path.join(args.out, 'train.csv'), index=False, header=False)
    val_df.to_csv(os.path.join(args.out, 'val.csv'), index=False, header=False)
    test_df.to_csv(os.path.join(args.out, 'test.csv'), index=False, header=False)

    for name, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rows = len(data)
        pulsars = data.iloc[:, -1].sum()
        pct = (pulsars / rows) * 100
        print(f"{name:<5}: {rows:>6} rows | pulsars: {pulsars:>5} ({pct:.2f}%)")

if __name__ == '__main__':
    main()
